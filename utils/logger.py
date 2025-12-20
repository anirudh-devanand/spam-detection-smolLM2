class avg_acc_logger:
    def __init__(self):
        self.total_samples = 0
        self.correct_samples = 0

    def update(self, is_correct):
        self.total_samples += len(is_correct)
        self.correct_samples += is_correct.sum().item()

    def compute_accuracy(self):
        if self.total_samples == 0:
            return 0.0
        return self.correct_samples / self.total_samples

class avg_logger:
    def __init__(self):
        self.total_count = 0
        self.total_value = 0.0

    def update(self, value, count=1):
        self.total_value += value * count
        self.total_count += count

    def compute_average(self):
        if self.total_count == 0:
            return 0.0
        return self.total_value / self.total_count