from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score

class MultiLabelModel:
    def __init__(self, algorithm='svm'):
        # Base estimator selection
        if algorithm == 'svm':
            base = LinearSVC(random_state=42, max_iter=10000)
        elif algorithm == 'rf':
            base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError("Algorithm must be 'svm' or 'rf'")
            
        # [5]: OneVsRest allows multi-label prediction
        self.clf = OneVsRestClassifier(base)
        
    def train(self, X, y):
        print("Training model...")
        self.clf.fit(X, y)
        print("Training complete.")
        
    def predict(self, X):
        return self.clf.predict(X)
        
    def evaluate(self, y_true, y_pred):
        """
        Computes multi-label metrics.[15, 16, 25]
        """
        return {
            'Accuracy (Subset)': accuracy_score(y_true, y_pred),
            'Hamming Loss': hamming_loss(y_true, y_pred),
            'F1 Score (Micro)': f1_score(y_true, y_pred, average='micro'),
            'F1 Score (Macro)': f1_score(y_true, y_pred, average='macro'),
            'Jaccard Score (Samples)': jaccard_score(y_true, y_pred, average='samples')
        }