from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # gives progression bars when running code

from activation_functions import logi, softmax
from data_loader import DataLoader
from loss_functions import mse_loss
from models import NeuralNetwork
from supplementary import Value, load_mnist

# Set printing precision for NumPy so that we don't get needlessly many digits in our answers.
np.set_printoptions(precision=2)

# Get images and corresponding labels from the (fashion-)mnist dataset
data_dir = Path(__file__).resolve().parent / "data"
train_images, train_y = load_mnist(data_dir, kind='train')
test_images, test_y = load_mnist(data_dir, kind='t10k')

# Reshape each of the 60 000 images from a 28x28 image into a 784 vector.
# Rescale the values in the 784 to be in [0,1] instead of [0, 255].
train_images = train_images.reshape(60_000, 784) / 255
test_images = test_images.reshape(10_000, 784) / 255

# Labels are stored as numbers. For neural network training, we want one-hot encoding, i.e. the label should be a vector
# of 10 long with a one in the index corresponding to the digit.
train_labels = np.zeros((60_000, 10))
train_labels[np.arange(60_000), train_y] = 1
test_labels = np.zeros((10_000, 10))
test_labels[np.arange(10_000), test_y] = 1

# We create our own validation set by placing the first 5000 images in the validation dataset and kepping the rest in
# the training set.
validation_subset = 5000
validation_images = train_images[:validation_subset]
validation_labels = train_labels[:validation_subset]
train_images = train_images[validation_subset:]
train_labels = train_labels[validation_subset:]

# The data loader takes at every iteration batch_size items from the dataset. If it is not possible to take batch_size
# items, it takes whatever it still can. With a dataset of 100 images and a batch size of 32, it will be batches of
# 32, 32, 32, and 4.
train_dataset = list(zip(train_images, train_labels))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
train_dataset_size = len(train_dataset)

validation_dataset = list(zip(validation_images, validation_labels))
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True, drop_last=False)
validation_dataset_size = len(validation_dataset)

test_dataset = list(zip(test_images, test_labels))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=False)
test_dataset_size = len(test_dataset)

# Initialize a neural network with some layers and the default activation functions.
neural_network = NeuralNetwork(
    layers=[784, 256, 128, 64, 10],
    activation_functions=[logi, logi, logi, softmax]
)
# OR load the parameters of some other trained network from disk
# neural_network = NeuralNetwork(
#   layers=[784, 256, 128, 64, 10],
#   activation_functions=[logi, logi, logi, softmax]
# ).load("path/to/some/folder")

# Set training configuration
learning_rate = 3e-3
epochs = 10

# Do the full training algorithm
train_losses = []
validation_losses = []
train_accuracies = []
validation_accuracies = []
for epoch in range(1, epochs+1):
    # (Re)set the training loss for this epoch.
    train_loss = 0.0
    correctly_classified = 0
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
        # Reset the gradients so that we start fresh.
        neural_network.reset_gradients()

        # Get the images and labels from the batch
        images = np.vstack([image for (image, _) in batch])
        labels = np.vstack([label for (_, label) in batch])

        # Wrap images and labels in a Value class.
        images = Value(images, expr="X")
        labels = Value(labels, expr="Y")

        # Compute what the model says is the label.
        output = neural_network(images)

        # Compute the loss for this batch.
        loss = mse_loss(
            output,
            labels
        )

        # Do backpropagation
        loss.backward()

        # Update the weights and biases using the chosen algorithm, in this case gradient descent.
        neural_network.gradient_descent(learning_rate)

        # Store the loss for this batch.
        train_loss += loss.data

        # Store accuracies for extra interpretability
        true_classification = np.argmax(
            labels.data,
            axis=1
        )
        predicted_classification = np.argmax(
            output.data,
            axis=1
        )
        correctly_classified += np.sum(true_classification == predicted_classification)

    # Store the loss and average accuracy for the entire epoch.
    train_losses.append(train_loss)
    train_accuracies.append(correctly_classified / train_dataset_size)

    print(f"Accuracy: {train_accuracies[-1]}")
    print(f"Loss: {train_loss}")
    print("")

    validation_loss = 0.0
    correctly_classified = 0
    for batch in tqdm(validation_loader, desc=f"Validation epoch {epoch}"):
        # Get the images and labels from the batch
        images = np.vstack([image for (image, _) in batch])
        labels = np.vstack([label for (_, label) in batch])

        # Wrap images and labels in a Value class.
        images = Value(images, expr="X")
        labels = Value(labels, expr="Y")

        # Compute what the model says is the label.
        output = neural_network(images)

        # Compute the loss for this batch.
        loss = mse_loss(
            output,
            labels
        )

        # Store the loss for this batch.
        validation_loss += loss.data

        # Store accuracies for extra interpretability
        true_classification = np.argmax(
            labels.data,
            axis=1
        )
        predicted_classification = np.argmax(
            output.data,
            axis=1
        )
        correctly_classified += np.sum(true_classification == predicted_classification)

    validation_losses.append(validation_loss)
    validation_accuracies.append(correctly_classified / validation_dataset_size)

    print(f"Accuracy: {validation_accuracies[-1]}")
    print(f"Loss: {validation_loss}")
    print("")

print(" === SUMMARY === ")
print(" --- training --- ")
print(f"Accuracies: {train_accuracies}")
print(f"Losses: {train_losses}")
print("")
print(" --- validation --- ")
print(f"Accuracies: {validation_accuracies}")
print(f"Losses: {validation_losses}")
print("")

# Plot of train vs test losses on the same axes
plt.figure()
plt.title("Loss: train vs validation")
plt.semilogy(np.array(range(1, epochs+1)), train_losses, label="train")
plt.semilogy(np.array(range(1, epochs+1)), validation_losses, label="validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot of train vs test loss on the x-axis but with different y-axis
figure, ax1 = plt.subplots()
color = "tab:blue"
ax1.set_title("Loss: train vs validation")
ax1.semilogy(np.array(range(1, epochs+1)), train_losses, color=color, label="train")
ax1.set_ylabel("Train loss", color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = "tab:orange"
ax2.semilogy(np.array(range(1, epochs+1)), validation_losses, color=color, label="validation")
ax2.set_ylabel("validation loss", color=color)
ax2.tick_params(axis='y', labelcolor=color)

figure.tight_layout()

# Plot of train vs test accuracies on the same axes
plt.figure()
plt.title("Accuracy: train vs validation")
plt.plot(np.array(range(1, epochs+1)), train_accuracies, label="train")
plt.plot(np.array(range(1, epochs+1)), validation_accuracies, label="validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot of train vs test accuracies on the x-axis but with different y-axis
figure, ax1 = plt.subplots()
color = "tab:blue"
ax1.set_title("Accuracy: train vs validation")
ax1.semilogy(np.array(range(1, epochs+1)), train_accuracies, color=color, label="train")
ax1.set_ylabel("Train accuracy", color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = "tab:orange"
ax2.semilogy(np.array(range(1, epochs+1)), validation_accuracies, color=color, label="validation")
ax2.set_ylabel("Test accuracy", color=color)
ax2.tick_params(axis='y', labelcolor=color)

figure.tight_layout()


# Compute the test loss and accuracies on the same axes
test_loss = 0.0
correctly_classified = 0
for batch in tqdm(test_loader, desc=f"Testing epoch {epoch}"):
    # Get the images and labels from the batch
    images = np.vstack([image for (image, _) in batch])
    labels = np.vstack([label for (_, label) in batch])

    # Wrap images and labels in a Value class.
    images = Value(images, expr="X")
    labels = Value(labels, expr="Y")

    # Compute what the model says is the label.
    output = neural_network(images)

    # Compute the loss for this batch.
    loss = mse_loss(
        output,
        labels
    )

    # Store the loss for this batch.
    test_loss += loss.data

    # Store accuracies for extra interpretability
    true_classification = np.argmax(
        labels.data,
        axis=1
    )
    predicted_classification = np.argmax(
        output.data,
        axis=1
    )
    correctly_classified += np.sum(true_classification == predicted_classification)

print(f"test loss:      {test_loss}")
print(f"test accuraccy: {correctly_classified / test_dataset_size}")

# We take a random starting point for 10 subsequent images we want to take a greater look at.
r = np.random.randint(0, 9_990)

# We go over 10 images starting with r, plot them and show the prediction the network makes next to them.
plt.figure()
for i in range(9):
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.subplot(3, 3, 1 + i)
    image = Value(np.array(test_images[r + i]), "x")
    plt.imshow(image.data.reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.text(-5, 45,
             f'True value:\n{test_labels[r + i]}: {test_y[r + i]}\n'
             f'Output:\n'
             f'[{neural_network(image)[0]:.2f} '  # needs __getitem__ method in Value class!
             f'{neural_network(image)[1]:.2f} '
             f'{neural_network(image)[2]:.2f} '
             f'{neural_network(image)[3]:.2f} '
             f'{neural_network(image)[4]:.2f}\n'
             f'{neural_network(image)[5]:.2f} '
             f'{neural_network(image)[6]:.2f} '
             f'{neural_network(image)[7]:.2f} '
             f'{neural_network(image)[8]:.2f} '
             f'{neural_network(image)[9]:.2f}]: {np.argmax(neural_network(image).data)}')

plt.subplots_adjust(hspace=.8)
plt.show()

# Save the parameters of the final network to disk
# neural_network.save("some_folder")
