"""
Card information database for Clash Royale
Contains detailed stats and categorization for all cards

Categories:
- 'melee': Close-range ground troops
- 'ranged': Ranged ground troops
- 'tank': High HP troops
- 'air': Flying troops
- 'building': Structures/spawners

Targets can be a list: ['ground'] or ['ground', 'air']
"""


CARD_INFO = {    'archers': {
        'category': 'ranged',
        'elixir_cost': 3,
        'targets': ['ground', 'air'],
        'range': 'long',
        'speed': 'medium',
    },
    'arrows': {
        'category': 'spell',
        'elixir_cost': 3,
        'targets': ['ground', 'air'],
        'range': 'spell',
        'speed': None,
    },
    'bomber': {
        'category': 'ranged',
        'elixir_cost': 2,
        'targets': ['ground'],
        'range': 'long',
        'speed': 'medium',
    },
    'knight': {
        'category': 'melee',
        'elixir_cost': 3,
        'targets': ['ground'],
        'range': 'melee',
        'speed': 'medium',
    },
    'fireball': {
        'category': 'spell',
        'elixir_cost': 4,
        'targets': ['ground', 'air'],
        'range': 'spell',
        'speed': None,
    },
    'mini_pekka': {
        'category': 'melee',
        'elixir_cost': 4,
        'targets': ['ground'],
        'range': 'melee',
        'speed': 'fast',
    },
    'musketeer': {
        'category': 'ranged',
        'elixir_cost': 4,
        'targets': ['ground', 'air'],
        'range': 'long',
        'speed': 'medium',
    },
    'giant': {
        'category': 'tank',
        'elixir_cost': 5,
        'targets': ['ground'],  # Only targets buildings
        'range': 'melee',
        'speed': 'slow',
    },
    'prince': {
        'category': 'melee',  # Could also be 'tank' due to high HP
        'elixir_cost': 5,
        'targets': ['ground'],
        'range': 'melee',
        'speed': 'medium',  # Fast when charging
    },
    'baby_dragon': {
        'category': 'air',
        'elixir_cost': 4,
        'targets': ['ground', 'air'],
        'range': 'medium',
        'speed': 'fast',
    },
    'skeleton_army': {
        'category': 'melee',
        'elixir_cost': 3,
        'targets': ['ground'],
        'range': 'melee',
        'speed': 'fast',
    },
    'witch': {
        'category': 'ranged',
        'elixir_cost': 5,
        'targets': ['ground', 'air'],
        'range': 'long',
        'speed': 'medium',
    },    
    'spear_goblins': {
        'category': 'ranged',
        'elixir_cost': 2,
        'targets': ['ground', 'air'],
        'range': 'long',
        'speed': 'very_fast',
    },
    'goblins': {
        'category': 'melee',
        'elixir_cost': 2,
        'targets': ['ground'],
        'range': 'melee',
        'speed': 'very_fast',
    },
    'skeletons': {
        'category': 'melee',
        'elixir_cost': 1,
        'targets': ['ground'],
        'range': 'melee',
        'speed': 'fast',
    },
    'minions': {
        'category': 'air',
        'elixir_cost': 3,
        'targets': ['ground', 'air'],
        'range': 'medium',
        'speed': 'fast',
    },
    'mega_minion': {
        'category': 'air',
        'elixir_cost': 3,
        'targets': ['ground', 'air'],
        'range': 'medium',
        'speed': 'medium',
    },
    'goblin_hut': {
        'category': 'building',
        'elixir_cost': 5,
        'targets': [],  # Spawner - doesn't attack
        'range': None,
        'speed': None,
    },    
    'valkyrie': {
        'category': 'melee',
        'elixir_cost': 4,
        'targets': ['ground'],
        'range': 'melee',
        'speed': 'medium',
    },
    'tombstone': {
        'category': 'building',
        'elixir_cost': 3,
        'targets': [],  
        'range': None,
        'speed': None,
    },
    'bomb_tower': {
        'category': 'building',
        'elixir_cost': 4,
        'targets': ['ground'],
        'range': 'medium',
        'speed': None,
    },
    'giant_skeleton': {
        'category': 'tank',
        'elixir_cost': 6,
        'targets': ['ground'],
        'range': 'melee',
        'speed': 'medium',
    },    
    'cannon': {
        'category': 'building',
        'elixir_cost': 3,
        'targets': ['ground'],
        'range': 'medium',
        'speed': None,
    },
    'barbarians': {
        'category': 'melee',
        'elixir_cost': 5,
        'targets': ['ground'],
        'range': 'melee',
        'speed': 'medium',
    },
    'rage': {
        'category': 'spell',
        'elixir_cost': 2,
        'targets': ['ground', 'air'],  # Affects all units
        'range': 'spell',
        'speed': None,
    },
    'rocket': {
        'category': 'spell',
        'elixir_cost': 6,
        'targets': ['ground', 'air'],
        'range': 'spell',
        'speed': None,
    },
    'barbarian_hut': {
        'category': 'building',
        'elixir_cost': 7,
        'targets': [],  # Spawner
        'range': None,
        'speed': None,
    },   
    'wizard': {
        'category': 'ranged',
        'elixir_cost': 5,
        'targets': ['ground', 'air'],
        'range': 'medium',
        'speed': 'medium',
    },
    'lightning': {
        'category': 'spell',
        'elixir_cost': 6,
        'targets': ['ground', 'air'],
        'range': 'spell',
        'speed': None,
    },
    'pekka': {
        'category': 'tank',
        'elixir_cost': 7,
        'targets': ['ground'],
        'range': 'melee',
        'speed': 'slow',
    },
    'hog_rider': {
        'category': 'melee',
        'elixir_cost': 4,
        'targets': ['ground'],  # Only targets buildings
        'range': 'melee',
        'speed': 'very_fast',
    },
    'minion_horde': {
        'category': 'air',
        'elixir_cost': 5,
        'targets': ['ground', 'air'],
        'range': 'medium',
        'speed': 'fast',
    },
    'x_bow': {
        'category': 'building',
        'elixir_cost': 6,
        'targets': ['ground'],
        'range': 'very_long',
        'speed': None,
    },    
    'battle_ram': {
        'category': 'tank',
        'elixir_cost': 4,
        'targets': ['ground'],  # Only targets buildings
        'range': 'melee',
        'speed': 'fast',
    },
    'fire_spirit': {
        'category': 'ranged',  # Jumps and deals splash damage
        'elixir_cost': 1,
        'targets': ['ground', 'air'],
        'range': 'melee',  # Has to jump to target
        'speed': 'very_fast',
    },
    'furnace': {
        'category': 'ranged',  # Reworked - now a troop
        'elixir_cost': 4,
        'targets': ['ground', 'air'],
        'range': 'medium',
        'speed': 'medium',
    },
    'goblin_cage': {
        'category': 'building',
        'elixir_cost': 4,
        'targets': ['ground'],
        'range': 'melee',
        'speed': None,
    },
    'skeleton_dragons': {
        'category': 'air',
        'elixir_cost': 4,
        'targets': ['ground', 'air'],
        'range': 'medium',
        'speed': 'fast',
    },
    'electro_spirit': {
        'category': 'ranged',
        'elixir_cost': 1,
        'targets': ['ground', 'air'],
        'range': 'melee',  # Jumps to target
        'speed': 'very_fast',
    },
    'inferno_tower': {
        'category': 'building',
        'elixir_cost': 5,
        'targets': ['ground', 'air'],
        'range': 'medium',
        'speed': None,
    },
}

def get_card_category(card_name: str) -> str:
    """
    Get card category (melee, ranged, tank, air, building, spell)

    Args:
        card_name: Name of the card (e.g., 'wizard')

    Returns:
        Category string, or 'melee' as default
    """
    return CARD_INFO.get(card_name, {}).get('category', 'melee')

def get_card_elixir(card_name: str) -> int:
    """
    Get elixir cost of a card

    Args:
        card_name: Name of the card

    Returns:
        Elixir cost (1-10), or 0 if unknown
    """
    return CARD_INFO.get(card_name, {}).get('elixir_cost', 0)

def get_card_targets(card_name: str) -> list:
    """
    Get what a card can target

    Args:
        card_name: Name of the card

    Returns:
        List of targets: ['ground'], ['air'], or ['ground', 'air']
    """
    return CARD_INFO.get(card_name, {}).get('targets', ['ground'])

def can_target_air(card_name: str) -> bool:
    """
    Check if a card can target air units

    Args:
        card_name: Name of the card

    Returns:
        True if card can hit air units
    """
    targets = get_card_targets(card_name)
    return 'air' in targets


# Backward compatibility: simple dictionary mapping card name -> category
# This is what CardHandDetector currently expects
CARD_TYPES = {
    card_name: info['category']
    for card_name, info in CARD_INFO.items()
}
