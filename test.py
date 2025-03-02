from mmengine.evaluator import BaseMetric
import torch


class OCRAccuracy(BaseMetric):
    """用于CRNN模型的文本准确率评估指标。

    计算预测文本与真实文本完全匹配的准确率。
    """

    def __init__(self, idx_to_char, collect_device="cpu", prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.idx_to_char = idx_to_char

    def process(self, data_batch, data_samples):
        """处理模型的输出和真实标签。

        Args:
            data_batch: 输入数据批次
            data_samples: 模型返回的输出，包括预测结果和标签
        """
        # 从模型输出中获取预测logits, 标签和长度信息
        log_probs, labels, target_lengths, texts = data_samples

        # 解码预测结果
        pred_strings = self.decode_predictions(log_probs)

        # 记录每个批次的正确预测数和总样本数
        correct = sum(
            1
            for pred_str, target_str in zip(pred_strings, texts)
            if pred_str == target_str
        )

        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({"batch_size": len(texts), "correct": correct})

    def compute_metrics(self, results):
        """计算整体评估指标。

        Args:
            results: 收集的所有批次结果

        Returns:
            dict: 包含评估指标的字典
        """
        total_correct = sum(item["correct"] for item in results)
        total_size = sum(item["batch_size"] for item in results)

        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(ocr_accuracy=100 * total_correct / total_size)

    def decode_predictions(self, log_probs):
        """解码模型的输出预测。

        Args:
            log_probs: 模型输出的对数概率，形状为[T, N, C]

        Returns:
            list: 解码后的文本列表
        """
        batch_size = log_probs.size(1)
        pred_sizes = torch.IntTensor([log_probs.size(0)] * batch_size)

        # 获取最高概率的字符索引
        _, preds = log_probs.max(2)
        preds = preds.transpose(1, 0).contiguous().cpu()

        # 解码预测结果
        pred_strings = []
        for i, pred in enumerate(preds):
            pred_string = ""
            prev_char = -1
            for p in pred[: pred_sizes[i]]:
                p = p.item()
                if p != 0 and p != prev_char:  # 不是空白符且不重复
                    pred_string += self.idx_to_char[p]
                prev_char = p
            pred_strings.append(pred_string)

        return pred_strings


class OCRCharAccuracy(BaseMetric):
    """字符级准确率评估指标。

    计算预测文本中每个字符的正确率，允许部分匹配。
    """

    def __init__(self, idx_to_char, collect_device="cpu", prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.idx_to_char = idx_to_char

    def process(self, data_batch, data_samples):
        """处理模型的输出和真实标签。"""
        log_probs, labels, target_lengths, texts = data_samples

        # 解码预测结果
        pred_strings = self.decode_predictions(log_probs)

        # 计算字符匹配情况
        total_chars = 0
        matched_chars = 0

        for pred_str, target_str in zip(pred_strings, texts):
            # 使用Levenshtein距离或其他方法计算字符匹配度
            # 这里简化为计算两个字符串的最长公共子序列长度
            lcs_len = self.longest_common_subsequence(pred_str, target_str)
            total_chars += len(target_str)
            matched_chars += lcs_len

        self.results.append(
            {"matched_chars": matched_chars, "total_chars": total_chars}
        )

    def compute_metrics(self, results):
        """计算整体评估指标。"""
        total_matched = sum(item["matched_chars"] for item in results)
        total_chars = sum(item["total_chars"] for item in results)

        return dict(char_accuracy=100 * total_matched / max(total_chars, 1))

    def decode_predictions(self, log_probs):
        """解码模型的输出预测。"""
        batch_size = log_probs.size(1)
        pred_sizes = torch.IntTensor([log_probs.size(0)] * batch_size)

        # 获取最高概率的字符索引
        _, preds = log_probs.max(2)
        preds = preds.transpose(1, 0).contiguous().cpu()

        # 解码预测结果
        pred_strings = []
        for i, pred in enumerate(preds):
            pred_string = ""
            prev_char = -1
            for p in pred[: pred_sizes[i]]:
                p = p.item()
                if p != 0 and p != prev_char:  # 不是空白符且不重复
                    pred_string += self.idx_to_char[p]
                prev_char = p
            pred_strings.append(pred_string)

        return pred_strings

    def longest_common_subsequence(self, str1, str2):
        """计算两个字符串的最长公共子序列长度。"""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]
