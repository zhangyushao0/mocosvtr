from mmengine.evaluator import BaseMetric
import torch


class OCRAccuracy(BaseMetric):
    """用于OCR模型的文本准确率评估指标。

    计算预测文本与真实文本完全匹配的准确率。
    """

    def __init__(self, dictionary, collect_device="cpu", prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.dictionary = dictionary

    def process(self, data_batch, data_samples):
        """处理模型的输出和真实标签。

        Args:
            data_batch: 输入数据批次，包含真实标签
            data_samples: 模型输出的预测概率(列表形式)
        """
        # 获取真实标签
        texts = [sample.gt_text.item for sample in data_batch['data_samples']]
        
        # 将列表形式的预测转换为张量进行处理
        # 假设data_samples中的每个元素是[T, C]形状的张量
        if isinstance(data_samples, list):
            # 先检查是否每个样本都是张量
            if all(isinstance(prob, torch.Tensor) for prob in data_samples):
                # 按照批次处理每个预测样本
                pred_strings = []
                for prob in data_samples:
                    # 这里假设prob的形状是[T, C]，需要根据实际情况处理
                    pred_str = self.decode_single_prediction(prob)
                    pred_strings.append(pred_str)
            else:
                # 如果不是张量列表，就尝试直接使用
                pred_strings = data_samples
        else:
            # 如果是单个张量，假设形状为[T, N, C]
            pred_strings = self.decode_predictions(data_samples)

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
        return dict(ocr_accuracy=100 * total_correct / max(total_size, 1))

    def decode_single_prediction(self, prob):
        """解码单个预测结果。

        Args:
            prob: 预测的logits，形状通常为[T, C]
                T: 序列长度, C: 类别数

        Returns:
            str: 解码后的文本
        """
        # 获取最高概率的字符索引
        _, pred = prob.max(1)
        pred = pred.cpu()

        # 解码预测结果
        pred_string = ""
        prev_char = -1
        for p in pred:
            p = p.item()
            # 跳过padding和重复字符
            if p != self.dictionary.padding_idx and p != prev_char:
                if 0 <= p < len(self.dictionary.dict):
                    pred_string += self.dictionary.dict[p]
            prev_char = p
        
        return pred_string

    def decode_predictions(self, probs):
        """解码模型的批量输出预测。

        Args:
            probs: 模型输出的概率，形状为[T, N, C]
                T: 序列长度, N: 批量大小, C: 类别数

        Returns:
            list: 解码后的文本列表
        """
        batch_size = probs.size(1)
        pred_sizes = torch.IntTensor([probs.size(0)] * batch_size)

        # 获取最高概率的字符索引
        _, preds = probs.max(2)
        preds = preds.transpose(1, 0).contiguous().cpu()  # [N, T]

        # 解码预测结果
        pred_strings = []
        for i, pred in enumerate(preds):
            pred_string = ""
            prev_char = -1
            for p in pred[: pred_sizes[i]]:
                p = p.item()
                # 跳过padding和重复字符
                if p != self.dictionary.padding_idx and p != prev_char:
                    if 0 <= p < len(self.dictionary.dict):
                        pred_string += self.dictionary.dict[p]
                prev_char = p
            pred_strings.append(pred_string)

        return pred_strings


class OCRCharAccuracy(BaseMetric):
    """字符级准确率评估指标。

    计算预测文本中每个字符的正确率，允许部分匹配。
    """

    def __init__(self, dictionary, collect_device="cpu", prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.dictionary = dictionary

    def process(self, data_batch, data_samples):
        """处理模型的输出和真实标签。"""
        # 获取真实标签
        texts = [sample.gt_text.item for sample in data_batch['data_samples']]
        
        # 将列表形式的预测转换为张量进行处理
        if isinstance(data_samples, list):
            # 先检查是否每个样本都是张量
            if all(isinstance(prob, torch.Tensor) for prob in data_samples):
                # 按照批次处理每个预测样本
                pred_strings = []
                for prob in data_samples:
                    pred_str = self.decode_single_prediction(prob)
                    pred_strings.append(pred_str)
            else:
                # 如果不是张量列表，就尝试直接使用
                pred_strings = data_samples
        else:
            # 如果是单个张量，假设形状为[T, N, C]
            pred_strings = self.decode_predictions(data_samples)

        # 计算字符匹配情况
        total_chars = 0
        matched_chars = 0

        for pred_str, target_str in zip(pred_strings, texts):
            # 使用最长公共子序列长度
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

    def decode_single_prediction(self, prob):
        """解码单个预测结果。"""
        # 获取最高概率的字符索引
        _, pred = prob.max(1)
        pred = pred.cpu()

        # 解码预测结果
        pred_string = ""
        prev_char = -1
        for p in pred:
            p = p.item()
            # 跳过padding和重复字符
            if p != self.dictionary.padding_idx and p != prev_char:
                if 0 <= p < len(self.dictionary.dict):
                    pred_string += self.dictionary.dict[p]
            prev_char = p
        
        return pred_string

    def decode_predictions(self, probs):
        """解码模型的批量输出预测。"""
        batch_size = probs.size(1)
        pred_sizes = torch.IntTensor([probs.size(0)] * batch_size)

        # 获取最高概率的字符索引
        _, preds = probs.max(2)
        preds = preds.transpose(1, 0).contiguous().cpu()

        # 解码预测结果
        pred_strings = []
        for i, pred in enumerate(preds):
            pred_string = ""
            prev_char = -1
            for p in pred[: pred_sizes[i]]:
                p = p.item()
                # 跳过padding和重复字符
                if p != self.dictionary.padding_idx and p != prev_char:
                    if 0 <= p < len(self.dictionary.dict):
                        pred_string += self.dictionary.dict[p]
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