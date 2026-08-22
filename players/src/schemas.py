from dataclasses import dataclass


@dataclass
class PlayerStats:
    id: int = 0
    points: int = 0
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    own_goals: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0
    influence: float = 0.0
    creativity: float = 0.0
    threat: float = 0.0
    ict_index: float = 0.0
    clearances_blocks_interceptions: int = 0
    recoveries: int = 0
    tackles: int = 0
    defensive_contribution: int = 0
    starts: int = 0
    xG: float = 0.0
    xA: float = 0.0
    xGI: float = 0.0
    xGc: float = 0.0
    value: float = 0.0
    selected: int = 0
    transfers_in: int = 0
    transfers_out: int = 0
