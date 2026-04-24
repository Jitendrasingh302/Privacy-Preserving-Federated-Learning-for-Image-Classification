import os
os.environ.pop('MPLBACKEND', None)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_metrics(fl_metrics, dp_fl_metrics, num_rounds, output_dir='.', output_filename='training_comparison.png'):
    """
    Plots the accuracy and loss curves for standard FL and DP-FL testing rounds.
    """
    rounds = range(1, num_rounds + 1)
    
    fl_acc = [m['accuracy'] for m in fl_metrics]
    dp_fl_acc = [m['accuracy'] for m in dp_fl_metrics]
    
    fl_loss = [m['loss'] for m in fl_metrics]
    dp_fl_loss = [m['loss'] for m in dp_fl_metrics]
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy curves
    plt.subplot(1, 2, 1)
    plt.plot(rounds, fl_acc, label='Standard FL', color='blue', marker='o')
    plt.plot(rounds, dp_fl_acc, label='DP-FL', color='orange', marker='s')
    plt.title('Client Accuracy vs Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss curves
    plt.subplot(1, 2, 2)
    plt.plot(rounds, fl_loss, label='Standard FL', color='blue', marker='o')
    plt.plot(rounds, dp_fl_loss, label='DP-FL', color='orange', marker='s')
    plt.title('Client Loss vs Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    print(f"Plots successfully saved to {output_path}")
    plt.close()
