from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
 
# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show()



#Evaluation (classification_report)
#classification_report(y_test, y_pred, target_names=...) prints precision, recall, F1-score per class plus overall averages, giving a fuller picture than accuracy alone.​
#“Support” in the report is the number of true test samples for each class, which helps interpret metrics (small support can make scores unstable).​#