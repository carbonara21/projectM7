import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import mediapipe as mp
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True)

expected_landmarks = 21

dataset_dir = "/Users/felipecarbone/PycharmProjects/Project/data"
categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
              "V", "W", "X", "Y", "Z"]

data, labels = [], []

for dir_ in os.listdir(dataset_dir):
    dir_path = os.path.join(dataset_dir, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            continue

        img = cv2.resize(img, (640, 480))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:  
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            if len(data_aux) == expected_landmarks * 2:  # Ensure correct number of landmarks (42 values)
                data.append(data_aux)
                labels.append(dir_)
        else:
            print(f"No landmarks detected in image {img_path}.")

x = np.asarray(data)
y = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=100)

# Train the base SVM model
base_model = SVC(kernel='linear', class_weight='balanced')
base_model.fit(x_train, y_train)
y_pred = base_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Base SVM Accuracy: {accuracy:.2f}")
print("Base SVM Precision:", precision_score(y_test, y_pred, average='macro'))
print("Base SVM Recall:", recall_score(y_test, y_pred, average='macro'))

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear'],
    'gamma': [0.0001, 0.001]
}

grid = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, verbose=3)
grid.fit(x_train, y_train)
best_params = grid.best_params_
best_model = grid.best_estimator_

y_pred_best = best_model.predict(x_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

print(f"Best SVM Accuracy: {accuracy_best:.2f}")
print("Best SVM Precision:", precision_score(y_test, y_pred_best, average='macro'))
print("Best SVM Recall:", recall_score(y_test, y_pred_best, average='macro'))

f = open('model.p', 'wb')
pickle.dump({'model': best_model}, f)
f.close()