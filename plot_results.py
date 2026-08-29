import matplotlib.pyplot as plt

# The evaluation data from your terminal
tracks = ['Spielberg\n(Trained Circuit)', 'Monza\n(Unseen Circuit)', 'Silverstone\n(Unseen Circuit)']
rewards = [6838.66, 741.84, 795.85]

# Set up the figure
plt.figure(figsize=(9, 6))

# Create a bar chart (Green for success, Red for failure)
bars = plt.bar(tracks, rewards, color=['#2ca02c', '#d62728', '#d62728'], edgecolor='black')

# Add titles and labels
plt.title('PPO Agent Zero-Shot Transfer Performance (2M Steps)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Total Episodic Reward', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Attach the exact numerical scores above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 100, 
             f'{yval:,.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Save the graph as an image for the slides
plt.tight_layout()
plt.savefig('generalization_results.png', dpi=300)
print("Graph saved as generalization_results.png!")

# Display the graph on your screen
plt.show()