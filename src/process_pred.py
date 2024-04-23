from generate_reasoning import ReasoningArgs, ReasoningGenerator


class ReasoningPredProcessor(ReasoningGenerator):
    def process_pred(self):
        with open(self.result_path, 'r', encoding='utf8')as f:
            preds = [line['reasoning'][1] for line in f.readlines()]
        