"""Home Assistant REST API client for occupancy detection.

Polls person.* entities via the HA REST API to determine if anyone is home.
Gracefully degrades: if HA is unreachable or not configured, assumes
everyone is home (conservative for consumption prediction).
"""

import logging
from datetime import datetime

import requests

import config
from storage.database import Database

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Polls HA person.* and climate.* entities for occupancy and AC state."""

    def __init__(self, db: Database):
        self.db = db
        self.enabled = config.homeassistant.enabled
        self.url = config.homeassistant.url.rstrip("/")
        self.token = config.homeassistant.token
        self.entities = [
            e.strip() for e in config.homeassistant.person_entities.split(",")
            if e.strip()
        ]
        self.climate_entities = [
            e.strip() for e in config.homeassistant.climate_entities.split(",")
            if e.strip()
        ]
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def poll_occupancy(self) -> bool:
        """Check if anyone is home. Returns True if any person entity is 'home'.

        Returns True (assume home) on any failure -- conservative for
        consumption prediction so we don't under-predict load.
        """
        if not self.enabled or not self.url or not self.entities:
            return True

        entity_states = {}
        anyone_home = False

        for entity_id in self.entities:
            try:
                resp = requests.get(
                    f"{self.url}/api/states/{entity_id}",
                    headers=self._headers,
                    timeout=10,
                )
                resp.raise_for_status()
                state = resp.json().get("state", "unknown")
                entity_states[entity_id] = state
                if state == "home":
                    anyone_home = True
            except requests.RequestException as e:
                logger.warning("HA entity %s unreachable: %s", entity_id, e)
                entity_states[entity_id] = "unknown"
                anyone_home = True  # fail-safe: assume home

        now = datetime.now().isoformat()
        self.db.insert_occupancy(now, anyone_home, entity_states)

        return anyone_home

    def poll_ac_state(self) -> bool:
        """Check if any AC/climate entity is actively cooling or heating.

        Returns False (AC off) if HA is unreachable or not configured.
        Climate entities report state as 'cool', 'heat', 'heat_cool',
        'dry', 'fan_only' when active, and 'off' when inactive.
        """
        if not self.enabled or not self.url or not self.climate_entities:
            return False

        for entity_id in self.climate_entities:
            try:
                resp = requests.get(
                    f"{self.url}/api/states/{entity_id}",
                    headers=self._headers,
                    timeout=10,
                )
                resp.raise_for_status()
                state = resp.json().get("state", "off")
                if state not in ("off", "unavailable", "unknown"):
                    return True
            except requests.RequestException as e:
                logger.warning("HA climate entity %s unreachable: %s", entity_id, e)

        return False
