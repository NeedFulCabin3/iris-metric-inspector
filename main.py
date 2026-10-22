import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    try:
        # 1. Load the Iris dataset
        print("Loading Iris dataset...")
        iris = load_iris()
        X, y = iris.data, iris.target
        target_names = iris.target_names

        # 2. Split data into training (80%) and testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 3. Initialize and train the K-Nearest Neighbors classifier
        print("Training K-Nearest Neighbors classifier...")
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        # 4. Evaluate the model on test data
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        # 5. Interactive terminal input loop for predictions
        print("\n--- Interactive Iris Species Predictor ---")
        print("Enter 4 flower measurements separated by commas or spaces.")
        print("Format: sepal_length, sepal_width, petal_length, petal_width")
        print("Type 'q' or 'quit' to exit.")

        while True:
            user_input = input("\nEnter measurements (or 'q' to quit): ").strip()
            
            if user_input.lower() in ("q", "quit"):
                print("Exiting application. Goodbye!")
                break

            try:
                # Parse user input into a list of floats (handling both commas and spaces)
                features = [float(x.strip()) for x in user_input.replace(",", " ").split()]

                if len(features) != 4:
                    print("Error: Please provide exactly 4 numerical values.")
                    continue

                # Make prediction
                sample = [features]
                prediction = model.predict(sample)
                predicted_species = target_names[prediction[0]]

                print(f"-> Predicted Species: {predicted_species.upper()}")

            except ValueError:
                print("Error: Invalid input. Please enter valid numbers (e.g., 5.1, 3.5, 1.4, 0.2).")

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()