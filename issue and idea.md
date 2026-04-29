async def stream_with_tool_calls(
    client_llm: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools_schema: List[Dict[str, Any]],
    available_tools: Dict[str, Callable],
    max_turns: int = _DEFAULT_MAX_TURNS,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Orchestrates the ReAct conversation using the primary LLM endpoint.

    Stream-friendly: answer text is yielded as `answer_chunk` events on
    channel "both" as it arrives. Reasoning is yielded as `reasoning_chunk`
    on channel "debug". Tool calls and tool results are also "debug".
    """
    extra_body = dict(extra_body or {})

    turn_count = 0
    is_last_turn = False
    last_function_name: Optional[str] = None

    while turn_count < max_turns:
        turn_count += 1
        logger.info(f"Turn {turn_count}/{max_turns} started.")

        is_last_turn = False

        try:
            yield _make_event("turn_start", {"turn_number": turn_count}, "debug")

            stream = await client_llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_schema if tools_schema else None,
                tool_choice="auto" if tools_schema else None,
                stream=True,
                extra_body=extra_body,
                **kwargs,
            )

            answer_accumulator = ""
            stripper = ThinkingStripper()
            tool_call_index_map: Dict[int, Dict[str, Any]] = {}
            in_reasoning = False  # Track native-thinking mode (Path A)

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Path A: provider exposes reasoning as a separate field.
                reasoning_delta = getattr(delta, "reasoning_content", None)
                if reasoning_delta:
                    in_reasoning = True
                    yield _make_event(
                        "reasoning_chunk", {"content": reasoning_delta}, "debug"
                    )
                    # DO NOT add reasoning_content to answer_accumulator — it
                    # must never leak into the user-facing answer.
                    continue

                # If we are currently in native-thinking mode, hold back any
                # delta.content until the thinking ends.
                if in_reasoning:
                    # The model switches from thinking to answer when
                    # reasoning_content stops arriving. We detect the end
                    # of thinking when content arrives but reasoning_content
                    # is empty (or we hit a tool call / stop).
                    # Qwen36 signals end-of-thinking with a special token or
                    # by simply stopping reasoning_content. For safety, we
                    # treat all content as answer once reasoning_content
                    # stops for a chunk that has content.
                    if delta.content:
                        in_reasoning = False
                        # Fall through to Path B to process this content.
                    else:
                        continue

                # Path B: reasoning is inline in content via <thinking> tags.
                if delta.content:
                    for channel, piece in stripper.feed(delta.content):
                        if channel == "answer":
                            answer_accumulator += piece
                            yield _make_event(
                                "answer_chunk", {"content": piece}, "both"
                            )
                        else:
                            yield _make_event(
                                "reasoning_chunk", {"content": piece}, "debug"
                            )

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        index = tc_delta.index
                        if index not in tool_call_index_map:
                            tool_call_index_map[index] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.id:
                            tool_call_index_map[index]["id"] += tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_call_index_map[index]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_call_index_map[index]["function"]["arguments"] += tc_delta.function.arguments

            # Drain any tail content the stripper held back.
            for channel, piece in stripper.flush():
                if channel == "answer":
                    answer_accumulator += piece
                    yield _make_event("answer_chunk", {"content": piece}, "both")
                else:
                    yield _make_event("reasoning_chunk", {"content": piece}, "debug")

            # Build the assistant message to feed back in. The content stored
            # in history is the answer-only portion (thinking is stripped).
            response_message: Dict[str, Any] = {
                "role": "assistant",
                "content": answer_accumulator if answer_accumulator else None,
            }

            tool_calls_list = list(tool_call_index_map.values())
            if tool_calls_list:
                response_message["tool_calls"] = tool_calls_list

            messages.append(response_message)

            if not tool_calls_list:
                # Model produced a final answer without a tool call.
                is_last_turn = True
                break

            # Build a per-tool-call summary with parsed arguments for consumers
            tool_call_summaries = []
            for tc in tool_calls_list:
                raw_args = tc.get("function", {}).get("arguments", "")
                parsed_args = {}
                if raw_args:
                    try:
                        parsed_args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {"_raw": raw_args}
                tool_call_summaries.append({
                    "call_id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": parsed_args,
                })

            yield _make_event(
                "tool_call",
                {
                    "tool_calls": tool_calls_list,
                    "tool_call_summaries": tool_call_summaries,
                    "turn": turn_count,
                },
                "debug",
            )

            # --- Execute the tools in parallel ---
            logger.info(f"Executing {len(tool_calls_list)} tool(s)...")

            async def execute_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
                function_name = tool_call["function"]["name"]
                call_id = tool_call["id"]
                start_time = time.time()

                tool_result_content = ""
                sources_for_debug: List[str] = []

                function_to_call = available_tools.get(function_name)
                if function_to_call:
                    try:
                        raw_args = tool_call["function"]["arguments"] or "{}"
                        try:
                            function_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            tool_result_content = "Error: Invalid JSON arguments generated by model."
                            function_args = {}

                        if not tool_result_content:
                            if asyncio.iscoroutinefunction(function_to_call):
                                response_data = await function_to_call(**function_args)
                            else:
                                response_data = await asyncio.to_thread(
                                    function_to_call, **function_args
                                )
                            tool_result_content, sources_for_debug = _unpack_tool_response(
                                response_data
                            )

                    except Exception as e:
                        logger.error(f"Error executing {function_name}: {e}", exc_info=True)
                        tool_result_content = f"System Error executing tool: {str(e)}"
                else:
                    tool_result_content = (
                        f"Error: Tool '{function_name}' not defined in available_tools_map."
                    )

                elapsed_ms = round((time.time() - start_time) * 1000)
                return {
                    "name": function_name,
                    "call_id": call_id,
                    "content": tool_result_content,
                    "sources": sources_for_debug,
                    "elapsed_ms": elapsed_ms,
                }

            results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls_list])

            for result in results:
                last_function_name = result["name"]
                yield _make_event(
                    "tool_result",
                    {
                        "call_id": result["call_id"],
                        "name": result["name"],
                        "sources": result["sources"],
                        "preview": result["content"],
                        "duration_ms": result["elapsed_ms"],
                        "status": "error" if result["content"].startswith("Error") else "ok",
                    },
                    "debug",
                )

                messages.append({
                    "tool_call_id": result["call_id"],
                    "role": "tool",
                    "name": result["name"],
                    "content": result["content"],
                })

        except BadRequestError as e:
            if "context length" in str(e).lower():
                logger.critical(
                    "Prompt exceeded model context window on turn %d. "
                    "Last tool=%s. Returning server-load fallback to user.",
                    turn_count,
                    last_function_name,
                )
                yield _make_event(
                    "answer_chunk", {"content": SERVER_LOAD_FALLBACK_BN}, "both"
                )
                raise ContextLengthExceededError() from e
            logger.error(f"API Bad Request: {e}")
            # Re-raise so orchestrator handles it (yields error + fallback)
            raise RuntimeError(f"API Bad Request: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in LLM loop: {e}", exc_info=True)
            yield _make_event(
                "error",
                {"content": "An internal error occurred.", "detail": str(e)},
                "debug",
            )
            yield _make_event(
                "answer_chunk", {"content": SERVER_LOAD_FALLBACK_BN}, "both"
            )
            raise

        yield _make_event("turn_end", {"turn_number": turn_count}, "debug")

    if not is_last_turn:
        logger.critical(
            "ReAct loop exhausted max_turns=%d without a final answer. "
            "Last tool=%s, partial answer length=%d chars.",
            max_turns,
            last_function_name,
            len(answer_accumulator) if 'answer_accumulator' in locals() else 0,
        )
        yield _make_event(
            "answer_chunk", {"content": SERVER_LOAD_FALLBACK_BN}, "both"
        )


after each turn a huge data dump is getting in history how to mange this? i.e its accumulating right? 
I want a system that can go to 50 turns even and wont run out 

once the model has decided on this turn can we summerize previous turns findings as summary?