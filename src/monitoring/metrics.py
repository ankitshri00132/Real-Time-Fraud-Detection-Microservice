class MetricsTracker:
    
    def __init__(self):

        self.total_request = 0
        self.total_prediction = 0
        self.total_fraud = 0
        self.error = 0

    def log_request(self):

        self.total_request += 1

    def log_prediction(self, prediction):

        self.total_prediction += int(len(prediction))
        self.total_fraud += int(sum(prediction))

    def log_error(self):
        self.error += 1

    def get_metrics(self):
        return {
            "total_requests": int(self.total_request),
            "total_predictions": int(self.total_prediction),
            "fraud_predictions": int(self.total_fraud),
            "total_errors": int(self.error)
        }