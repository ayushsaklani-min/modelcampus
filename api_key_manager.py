"""
API Key Manager - Generate and manage API keys
"""

import secrets
import json
import os
from datetime import datetime

class APIKeyManager:
    """Manage API keys for your API"""
    
    def __init__(self, keys_file='api_keys.json'):
        self.keys_file = keys_file
        self.keys = self._load_keys()
    
    def _load_keys(self):
        """Load API keys from file"""
        if os.path.exists(self.keys_file):
            with open(self.keys_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_keys(self):
        """Save API keys to file"""
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, indent=2)
    
    def generate_key(self, name='', rate_limit=100, expires_days=None):
        """
        Generate a new API key
        
        Args:
            name: Name/description for the key
            rate_limit: Requests per day
            expires_days: Days until expiration (None = never expires)
        
        Returns:
            str: Generated API key
        """
        api_key = secrets.token_urlsafe(32)
        
        key_data = {
            'name': name or f'Key {datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'created_at': datetime.now().isoformat(),
            'rate_limit': rate_limit,
            'requests_today': 0,
            'last_reset': datetime.now().isoformat(),
            'active': True
        }
        
        if expires_days:
            from datetime import timedelta
            key_data['expires_at'] = (datetime.now() + timedelta(days=expires_days)).isoformat()
        
        self.keys[api_key] = key_data
        self._save_keys()
        
        return api_key
    
    def validate_key(self, api_key):
        """Validate if API key exists and is active"""
        if api_key not in self.keys:
            return False, 'Invalid API key'
        
        key_data = self.keys[api_key]
        
        if not key_data.get('active', True):
            return False, 'API key is inactive'
        
        # Check expiration
        if 'expires_at' in key_data:
            expires_at = datetime.fromisoformat(key_data['expires_at'])
            if datetime.now() > expires_at:
                return False, 'API key has expired'
        
        # Check rate limit
        if key_data.get('requests_today', 0) >= key_data.get('rate_limit', 100):
            return False, 'Rate limit exceeded'
        
        return True, key_data
    
    def increment_usage(self, api_key):
        """Increment request count for API key"""
        if api_key in self.keys:
            key_data = self.keys[api_key]
            last_reset = datetime.fromisoformat(key_data.get('last_reset', datetime.now().isoformat()))
            
            # Reset daily counter if new day
            if (datetime.now() - last_reset).days >= 1:
                key_data['requests_today'] = 0
                key_data['last_reset'] = datetime.now().isoformat()
            
            key_data['requests_today'] = key_data.get('requests_today', 0) + 1
            self._save_keys()
    
    def revoke_key(self, api_key):
        """Revoke/deactivate an API key"""
        if api_key in self.keys:
            self.keys[api_key]['active'] = False
            self._save_keys()
            return True
        return False
    
    def list_keys(self):
        """List all API keys (without showing full keys)"""
        return {
            key[:10] + '...': {
                'name': data.get('name'),
                'active': data.get('active'),
                'rate_limit': data.get('rate_limit'),
                'requests_today': data.get('requests_today', 0),
                'created_at': data.get('created_at')
            }
            for key, data in self.keys.items()
        }


# CLI tool to manage keys
if __name__ == '__main__':
    import sys
    
    manager = APIKeyManager()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python api_key_manager.py generate [name] [rate_limit]")
        print("  python api_key_manager.py list")
        print("  python api_key_manager.py revoke <key_prefix>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'generate':
        name = sys.argv[2] if len(sys.argv) > 2 else ''
        rate_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        
        key = manager.generate_key(name=name, rate_limit=rate_limit)
        print(f"\n[SUCCESS] Generated API Key:")
        print(f"   {key}")
        print(f"\n[WARNING] Save this key securely! It will not be shown again.")
        print(f"\nUsage:")
        print(f"   curl -H 'X-API-Key: {key}' http://localhost:5000/predict")
    
    elif command == 'list':
        keys = manager.list_keys()
        print("\nAPI Keys:")
        for key_prefix, info in keys.items():
            status = "[ACTIVE]" if info['active'] else "[INACTIVE]"
            print(f"  {key_prefix} - {info['name']} - {status}")
            print(f"    Rate Limit: {info['rate_limit']}/day, Used: {info['requests_today']}")
    
    elif command == 'revoke':
        if len(sys.argv) < 3:
            print("Error: Provide key prefix to revoke")
            sys.exit(1)
        
        prefix = sys.argv[2]
        # Find matching key
        for key in manager.keys:
            if key.startswith(prefix):
                manager.revoke_key(key)
                print(f"[SUCCESS] Revoked key: {key[:20]}...")
                break
        else:
            print("[ERROR] Key not found")
    
    else:
        print(f"Unknown command: {command}")

