#!/bin/bash

cd /home/ai/Workspace/Rafik/QA_System-main
source venv/bin/activate

echo "🔄 Refreshing CRM token..."
python -c "from app.service_hub.crm_doctors import fetch_doctors; fetch_doctors(force_refresh=True)"

echo ""
echo "📋 Copying token to container data directory..."
cp /home/ai/.andalusia_crm_cache.bin data/.andalusia_crm_cache.bin

echo ""
echo "⏰ Checking token expiration..."
python3 -c "
import json
import datetime

with open('data/.andalusia_crm_cache.bin', 'r') as f:
    data = json.load(f)
    token = list(data['AccessToken'].values())[0]
    expires = int(token['expires_on'])
    exp_dt = datetime.datetime.fromtimestamp(expires, datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    remaining = expires - int(now.timestamp())
    
    print(f'Token expires:   {exp_dt.strftime(\"%Y-%m-%d %H:%M:%S UTC\")}')
    print(f'Current time:    {now.strftime(\"%Y-%m-%d %H:%M:%S UTC\")}')
    print(f'Time remaining:  {remaining // 3600}h {(remaining % 3600) // 60}m')
    print(f'Status:          {\"✅ VALID\" if remaining > 0 else \"❌ EXPIRED\"}')
"

echo ""
echo "🔄 Restarting container..."
docker compose restart api

echo ""
echo "✅ Done! Token refreshed and container restarted."