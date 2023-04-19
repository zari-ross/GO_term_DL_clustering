import matplotlib.pyplot as plt

accuracy = 'classification_output_accuracy'
train_acc = {('%s' % accuracy): []}
val_acc = {('val_%s' % accuracy): []}

# Load the training log file
with open('GO_term_training.log', 'r') as file:
    lines = file.readlines()
    for line_number, line in enumerate(lines[1:], start=0):
        values = line.split(",")
        epoch = line_number + 1
        train_acc[accuracy].append(float(values[1]))

# Create the plot
epochs = range(1, len(lines))
plt.plot(epochs, train_acc[accuracy], label='Train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
