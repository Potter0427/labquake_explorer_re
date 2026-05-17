import json
from pathlib import Path

PREFS_FILE = Path.home() / ".labquake_explorer_prefs.json"

class UserPrefs:
    _prefs = None

    @classmethod
    def load(cls):
        if cls._prefs is None:
            try:
                if PREFS_FILE.exists():
                    with open(PREFS_FILE, 'r', encoding='utf-8') as f:
                        cls._prefs = json.load(f)
                else:
                    cls._prefs = {}
            except Exception as e:
                print(f"Failed to load user prefs: {e}")
                cls._prefs = {}
        return cls._prefs

    @classmethod
    def save(cls):
        if cls._prefs is not None:
            try:
                with open(PREFS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(cls._prefs, f, indent=4)
            except Exception as e:
                print(f"Failed to save user prefs: {e}")

    @classmethod
    def get(cls, view_name, key, default=None):
        prefs = cls.load()
        if view_name not in prefs:
            return default
        return prefs[view_name].get(key, default)

    @classmethod
    def set(cls, view_name, key, value):
        prefs = cls.load()
        if view_name not in prefs:
            prefs[view_name] = {}
        prefs[view_name][key] = value
        cls.save()
