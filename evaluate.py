import gymnasium as gym
import f1tenth_gym
from stable_baselines3 import PPO
from sb3_wrapper import F1TenthSB3Wrapper
import matplotlib.pyplot as plt

def main():
    print("Loading saved PPO model...")
    # Make sure this matches the filename you saved in train.py!
    model = PPO.load("ppo_f1tenth_2M_model")

    test_tracks = ["Spielberg", "Monza", "Silverstone"] 
    final_rewards = []

    # 1. Run the evaluations
    for track_name in test_tracks:
        print(f"\n--- Initializing Environment for track: {track_name} ---")
        
        env = gym.make("f1tenth_gym:f1tenth-v0", config={
            "num_agents": 1,
            "timestep": 0.01,
            "map": track_name  
        }, render_mode="human")
        
        env = F1TenthSB3Wrapper(env)
        
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            
        print(f"Finished {track_name} | Total Reward = {total_reward:.2f}")
        final_rewards.append(total_reward)
        env.close()

    # 2. Generate the results graph automatically
    print("\nGenerating results graph...")
    formatted_tracks = [f'{t}\n(Trained)' if t == 'Spielberg' else f'{t}\n(Unseen)' for t in test_tracks]
    
    plt.figure(figsize=(9, 6))
    
    # Color the trained track green, and unseen tracks red
    colors = ['#2ca02c' if 'Trained' in t else '#d62728' for t in formatted_tracks]
    bars = plt.bar(formatted_tracks, final_rewards, color=colors, edgecolor='black')

    plt.title('PPO Agent Zero-Shot Transfer Performance', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Total Episodic Reward', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(final_rewards)*0.02), 
                 f'{yval:,.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('generalization_results.png', dpi=300)
    print("Success! Graph saved as generalization_results.png")
    plt.show()

if __name__ == "__main__":
    main()