import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display/generate_metrics/roc_auc")
mode = ap.parse_args().mode

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surpris"}

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 classes for emotions

# If mode is "train", train the model
if mode == "train":
    train_dir = 'data/train'
    val_dir = 'data/test'

    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, epsilon=1e-6), metrics=['accuracy'])
    
    # Train the model and store the training history
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    
    model.save_weights('model.weights.h5')

    # Plot Accuracy and Loss Curves
    def plot_model_history(history):
        """
        Plot training & validation accuracy and loss values
        """
        # Create a figure for accuracy and loss plots
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
        plt.title('Model Accuracy for All Emotions')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True)

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss', color='red')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
        plt.title('Model Loss for All Emotions')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig('accuracy_loss_curves_all_emotions.png')
        plt.show()

    # Call the function to plot accuracy and loss
    plot_model_history(model_info)

# If mode is "display", display the emotions on webcam
elif mode == "display":
    model.load_weights('model.h5')
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(0)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (800, 600), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

# New mode to generate precision and confusion matrix
elif mode == "generate_metrics":
    model.load_weights('model.h5')
    
    # Load the validation dataset
    val_dir = 'data/test'
    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)
    
    # Get predictions
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate precision
    class_precision = precision_score(y_true, y_pred, average=None)
    
    emotion_labels = list(emotion_dict.values())
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Plot precision for each class
    plt.figure(figsize=(8, 6))
    plt.bar(emotion_labels, class_precision)
    plt.ylim(0, 1)
    plt.ylabel('Precision')
    plt.title('Precision for Each Emotion')
    plt.savefig('precision_per_class.png')
    plt.show()
    
    # Print the precision for each class
    for label, precision in zip(emotion_labels, class_precision):
        print(f"{label}: {precision:.4f}")

# New mode to generate ROC curves and AUC for each class
elif mode == "roc_auc":
    model.load_weights('model.h5')
    
    # Load the validation dataset
    val_dir = 'data/test'
    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)
    
    # Get true labels and predicted probabilities
    y_true = validation_generator.classes
    y_score = model.predict(validation_generator, steps=validation_generator.samples // validation_generator.batch_size + 1)
    
    # Binarize the labels for multi-class ROC computation
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6])
    n_classes = 7  # Number of classes

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of {emotion_dict[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Emotion')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.show()

    # Print AUC for each class
    for i, label in emotion_dict.items():
        print(f'{label}: AUC = {roc_auc[i]:0.4f}')
