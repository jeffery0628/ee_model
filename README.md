# plmee
plmee事件抽取用于裁判文书事件抽取

## 触发器的抽取
触发器抽取器的目的是预测出触发了事件的token，形式化为token级别的多类别分类任务，分类标签是事件类型。在BERT上添加一个多类分类器就构成了触发器抽取器。

触发器抽取器的输入和BERT的一样，是WordPiece嵌入、位置嵌入和segment嵌入的和。因为输入只有一个句子，所以所有的segment ids设为0。句子首尾的token分别是[CLS]和[SEP]。

触发词有时不是一个单词而是一个词组。因此，作者令连续的tokens共享同一个预测标签，作为一个完整的触发词。

采用交叉熵损失函数用于微调（fine-tune）。

![](images/image-20200626162628002.png)

### 模型

```python
class TriggerExtractor(BaseModel):
    def __init__(self, bert_path, bert_train, dropout,event_type_num):
        super(TriggerExtractor,self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # 对bert进行训练
        for name, param in self.bert.named_parameters():
            param.requires_grad = bert_train
        self.dropout = nn.Dropout(dropout)
        self.taggers = nn.ModuleList([nn.Linear(self.bert.config.to_dict()['hidden_size'],2) for i in range(event_type_num)])

    def forward(self, text_lengths, text_ids, masks):
        bert_out,bert_cls = self.bert(text_ids,attention_mask=masks)
        bert_out = self.dropout(bert_out)
        outputs = []
        for tagger in self.taggers:
            out = tagger(bert_out).unsqueeze(-2)
            outputs.append(out[:,1:,:])
        return torch.cat(outputs,dim=-2)
```

### 评测指标

1. 位置正确
2. 位置正确且类别正确

## 元素的抽取
给定触发器的条件下，元素抽取器的目的是抽取出和触发器所对应事件相关的元素，以及这些元素扮演的角色。

和触发器抽取相比较，元素的抽取更加复杂，主要有3个原因：

1. 元素对触发器的依赖；
2. 大多数元素是较长的名词短语；
3. 角色重叠问题。

和触发器抽取器一样，元素抽取器也需要3种嵌入相加作为输入，但还需要知道哪些tokens组成了触发器，因此特征表示输入的segment将触发词所在的span设为1。

为了克服元素抽取面临的后2个问题，作者在BERT上添加了多组二类分类器（多组分类器设为所有角色标签的集合，对每个元素判断所有类型角色的概率）。每组分类器服务于一个角色，以确定所有属于它的元素的范围（span：每个span都包括start和end）。

由于预测和角色是分离的，所以一个元素可以扮演多个角色（对一个元素使用一组二类分类器，一组二类分类器中有多个分类器，对应多个角色），一个token可以属于不同的元素。这就缓解了角色重叠问题。


```python
class ArgumentExtractor(BaseModel):
    def __init__(self, bert_path, bert_train, role_type_num, dropout):
        super(ArgumentExtractor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # 对bert进行训练
        for name, param in self.bert.named_parameters():
            param.requires_grad = bert_train

        self.start_taggers = nn.ModuleList(
            [nn.Linear(self.bert.config.to_dict()['hidden_size'], 2) for i in range(role_type_num)])
        self.end_taggers = nn.ModuleList(
            [nn.Linear(self.bert.config.to_dict()['hidden_size'], 2) for i in range(role_type_num)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_lengths, text_ids, masks, type_ids):
        bert_out, bert_cls = self.bert(text_ids, attention_mask=masks, token_type_ids=type_ids)
        bert_out = self.dropout(bert_out)
        start_outputs = []
        end_outputs = []
        for start_tagger, end_tagger in zip(self.start_tagger, self.end_taggers):
            start_out = start_tagger(bert_out).unsqueeze(-2)
            end_out = end_tagger(bert_out).unsqueeze(-2)
            start_outputs.append(start_out[:, 1:, :])
            end_outputs.append(end_out[:, 1:, :])
        return torch.cat(start_outputs, dim=-2), torch.cat(end_outputs, dim=-2)
```

### 评测指标

1. 位置正确
2. 位置正确且类别正确

## AF-IEF

Role Frequency （RF）
RF定义为角色r在类型为v的事件中出现的频率：
$$
\operatorname{RF}(r, v)=\frac{N_{v}^{r}}{\sum_{k \in \mathcal{R}} N_{v}^{k}}
$$
Inverse Event Frequency （IEF）

log内的分母表示论元角色r在多少个事件从出现
$$
\operatorname{IEF}(r)=\log \frac{|\mathcal{V}|}{|\{v \in \mathcal{V}: r \in v\}|}
$$

```python
def compute_AF_IEF(self, dataset):
        event_num = len(self.schema.event_type_2_id)
        role_num = len(self.schema.role_type_2_id)

        self.rf = np.zeros((event_num, role_num), dtype=float)
        self.ief = np.zeros((role_num, ), dtype=float)
		
         with open(os.path.join(self.data_dir, file), 'r') as f:
             for line in f:
                json_item = json.loads(line)
                for event in json_item['events'].values():
                    event_id = self.schema.event_type_2_id[event['event_type']]

                    for argu in event['argument']:
                        role_id = self.schema.role_type_2_id[argu['role_type']]
                        self.rf[event_id, role_id] += 1

                for idx in range(role_num):
                    self.ief[idx] = np.log(event_num/np.sum(self.af[:,idx] != 0))
                role_count_per_event = np.sum(self.rf, axis=1)
                self.rf = self.rf / (role_count_per_event+1e-13)
```

将$RF(r,v)$和$IEF(r)$相乘得到$RF−IEF(r,v)$，使用RF-IEF度量角色r对于v类型事件的重要性：

$$
I(r, v)=\frac{\exp ^{\operatorname{RF}-\operatorname{IEF}}(r, v)}{\sum_{r^{\prime} \in R} \exp ^{\operatorname{RF}-\operatorname{IEF}}\left(r^{\prime}, v\right)}
$$
给定输入的事件类型$v$，根据每个角色对于$v$类型事件的重要性，计算损失$L_s$和$L_e$ ，将两者取平均就得到最终的损失。
$$
\begin{aligned}
\mathcal{L}_{s} &=\sum_{r \in \mathcal{R}} \frac{I(r, v)}{|\mathcal{S}|} \mathrm{CE}\left(P_{s}^{r}, \boldsymbol{y}_{s}^{r}\right) \\
\mathcal{L}_{e} &=\sum_{r \in \mathcal{R}} \frac{I(r, v)}{|\mathcal{S}|} \mathrm{CE}\left(P_{e}^{r}, \boldsymbol{y}_{e}^{r}\right)
\end{aligned}
$$

```python
rf_ief = ief * rf
batch_weight = torch.exp(rf_ief)
batch_weight_sum = torch.sum(batch_weight)
batch_weight = batch_weight / batch_weight_sum
batch_loss = batch_loss * batch_weight
```

