"""
Example: How to integrate Tower HP detection into your RL training loop

This is a reference showing EXACTLY where to add tower HP detection
when you implement RL training.
"""

from detection.tower_hp_detector import TowerHPDetector
from rl.state_encoder import StateEncoder
from detection.card_hand_detector import CardHandDetector
from detection.elixir_detector import ElixirDetector
from controllers.game_controller import GameController
import time


def rl_training_loop_example():
    """
    Example RL training loop with tower HP integrated
    """
    # ===== INITIALIZATION =====
    gc = GameController(instance_id=0)
    
    # Add tower HP detector
    tower_hp_detector = TowerHPDetector()
    
    # Other detectors
    hand_detector = CardHandDetector()
    elixir_detector = ElixirDetector()
    state_encoder = StateEncoder(grid_rows=32, grid_cols=18, max_cards=4)
    
    # Tower HP caching (for performance)
    cached_tower_hp = None
    last_tower_hp_update = 0
    TOWER_HP_UPDATE_INTERVAL = 2.0  # Update every 2 seconds
    
    # Your RL agent
    # rl_agent = YourRLAgent(state_size=state_encoder.state_size, action_size=...)
    
    # ===== BATTLE LOOP =====
    while in_battle:
        screenshot = gc.take_screenshot()
        
        # 1. Detect elixir (every frame - fast)
        elixir = elixir_detector.get_elixir(screenshot, verbose=False)
        
        # 2. Detect hand (every frame - fast)
        hand = hand_detector.get_hand(screenshot, verbose=False)
        
        # 3. Detect tower HP (throttled - OCR is slower)
        current_time = time.time()
        if current_time - last_tower_hp_update > TOWER_HP_UPDATE_INTERVAL:
            cached_tower_hp = tower_hp_detector.get_tower_hp(screenshot, verbose=False)
            last_tower_hp_update = current_time
        
        # Use cached tower HP if recent update exists
        tower_hp = cached_tower_hp if cached_tower_hp is not None else {
            'enemy_left_princess': 0,
            'enemy_king': 0,
            'enemy_right_princess': 0,
            'ally_left_princess': 0,
            'ally_king': 0,
            'ally_right_princess': 0,
        }
        
        # 4. Detect troops (if using YOLO)
        all_detections = []  # Your troop detection here
        enemy_detections = [d for d in all_detections if d['class_name'].startswith('enemy')]
        ally_detections = [d for d in all_detections if d['class_name'].startswith('ally')]
        
        # 5. Encode state with tower HP
        state = state_encoder.encode_state(
            elixir=elixir if elixir is not None else 5.0,
            hand=hand,
            enemy_detections=enemy_detections,
            ally_detections=ally_detections,
            tower_hp=tower_hp  # <-- TOWER HP INTEGRATED HERE
        )
        
        # 6. RL agent selects action
        # action = rl_agent.select_action(state)
        # card_slot, row, col = decode_action(action)
        
        # 7. Execute action
        # gc.play_card(card_slot, row, col)
        
        # 8. Calculate reward (can use tower HP changes!)
        # reward = calculate_reward(prev_tower_hp, tower_hp)
        
        # 9. Store experience for training
        # rl_agent.store_experience(state, action, reward, next_state, done)
        
        # 10. Train agent
        # if enough_experience:
        #     rl_agent.train()
        
        time.sleep(0.1)  # Small delay


def calculate_reward_with_tower_hp(prev_tower_hp, current_tower_hp):
    """
    Example reward function using tower HP
    
    Rewards:
    - +0.01 per damage to enemy towers
    - +10.0 for destroying enemy tower
    - -0.01 per damage to ally towers
    - -10.0 for losing ally tower
    """
    reward = 0.0
    
    # Enemy tower damage = positive
    for tower in ['enemy_left_princess', 'enemy_king', 'enemy_right_princess']:
        prev_hp = prev_tower_hp.get(tower, 0)
        curr_hp = current_tower_hp.get(tower, 0)
        damage = prev_hp - curr_hp
        
        if damage > 0:
            reward += damage * 0.01
            
            # Bonus for destroying tower
            if prev_hp > 0 and curr_hp == 0:
                reward += 10.0
    
    # Ally tower damage = negative
    for tower in ['ally_left_princess', 'ally_king', 'ally_right_princess']:
        prev_hp = prev_tower_hp.get(tower, 0)
        curr_hp = current_tower_hp.get(tower, 0)
        damage = prev_hp - curr_hp
        
        if damage > 0:
            reward -= damage * 0.01
            
            # Penalty for losing tower
            if prev_hp > 0 and curr_hp == 0:
                reward -= 10.0
    
    return reward


if __name__ == "__main__":
    print("This is a reference example showing tower HP integration.")
    print("Copy the relevant parts into your RL training code.")
    print("\nKey points:")
    print("1. Initialize TowerHPDetector()")
    print("2. Update tower HP every 2-3 seconds (throttled)")
    print("3. Pass tower_hp to state_encoder.encode_state()")
    print("4. Optionally use tower HP changes for reward shaping")
