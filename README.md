# CogOpsCB
# Creating and Managing a systemd Service (`govtchat.service`)

Here's a complete guide to deploying your `govtchat.service` file correctly.

## 1. Service File Location

Copy your service file to the appropriate directory based on scope:

| Scope | Path | Use Case |
|-------|------|----------|
| **System-wide** (recommended) | `/etc/systemd/system/govtchat.service` | Runs as root/system user; starts at boot |
| **Per-user** | `~/.config/systemd/user/govtchat.service` | Runs under your user account; requires `systemctl --user` |

> ✅ **Recommendation**: Use `/etc/systemd/system/` for production services.

```bash
sudo cp govtchat.service /etc/systemd/system/
```

## 2. Set Correct Permissions

```bash
sudo chmod 644 /etc/systemd/system/govtchat.service
```

## 3. Reload systemd Daemon

Always reload after adding/modifying service files:

```bash
sudo systemctl daemon-reload
```

## 4. Enable & Start the Service

```bash
# Enable to start on boot
sudo systemctl enable govtchat.service

# Start immediately
sudo systemctl start govtchat.service
```

## 5. Verify Status

```bash
# Check if running
sudo systemctl status govtchat.service

# View recent logs
sudo journalctl -u govtchat.service -f  # -f for follow/live logs
```

## 6. Example Service File Template

```ini
[Unit]
Description=GovtChat Service
After=network.target

[Service]
Type=simple
User=www-data          # Run as non-root user (recommended)
WorkingDirectory=/opt/govtchat
ExecStart=/opt/govtchat/govtchat --port=8080
Restart=on-failure
RestartSec=5s
Environment="ENV=production"

# Security hardening (optional but recommended)
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

## 7. Best Practices Checklist

- ✅ **Run as non-root user** (`User=` directive)
- ✅ **Set `WorkingDirectory`** to your app's root
- ✅ **Use absolute paths** in `ExecStart`
- ✅ **Validate syntax** before reloading:
  ```bash
  sudo systemd-analyze verify /etc/systemd/system/govtchat.service
  ```
- ✅ **Test manually first**:
  ```bash
  sudo -u www-data /opt/govtchat/govtchat --port=8080
  ```

## 8. Common Commands Reference

| Command | Description |
|---------|-------------|
| `sudo systemctl stop govtchat.service` | Stop service |
| `sudo systemctl restart govtchat.service` | Restart service |
| `sudo systemctl disable govtchat.service` | Disable auto-start |
| `sudo journalctl -u govtchat.service --since "5 min ago"` | View recent logs |
| `sudo systemctl cat govtchat.service` | Show loaded service config |

## 9. Troubleshooting

- **"Unit not found" after copy?** → Forgot `daemon-reload`
- **Permission denied?** → Check file permissions (`644`) and `User=` directive
- **Service fails to start?** → Check logs: `journalctl -u govtchat.service -b`
- **Path errors?** → Always use absolute paths in `ExecStart` and `WorkingDirectory`

> 💡 **Pro Tip**: Test your service manually first before enabling it as a systemd unit to catch runtime errors early.