import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('learning_curve.csv')

# Drop rows where either eval_loss or eval_accuracy is NaN
data = data.dropna(subset=['eval_loss', 'eval_accuracy'])

# Extract data
epochs = data.index  # Assuming the index represents epochs
eval_loss = data['eval_loss']
eval_accuracy = data['eval_accuracy']

# Plot the data
plt.figure(figsize=(10, 6))
plt.xlabel('Epoch', fontsize=12)
# Plot eval_loss on secondary y-axis
plt.plot(epochs,
         eval_loss,
         marker='o',
         linestyle='-',
         color='b',
         label='Evaluation Loss')
plt.ylabel('Loss', color='b')

# Create a twin Axes sharing the xaxis
plt.twinx()

# Plot eval_accuracy on primary y-axis
plt.plot(epochs,
         eval_accuracy,
         marker='o',
         linestyle='-',
         color='g',
         label='Evaluation Accuracy')
plt.ylabel('Accuracy', color='g')

plt.title('Evaluation Metrics')

plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig('report_data.png')

# Display the plot
plt.show()
