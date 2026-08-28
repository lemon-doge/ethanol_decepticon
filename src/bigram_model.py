import torch
import torch.nn as nn
from data_loader import DataLoader
from tokenizer import CharTokenizer
from settings import device, seed

## simplest bigram model - predict next token based on the current token only

## hyperparams
torch.manual_seed(seed)

batch_size = 32
context_len = 1
lr = 1e-3
train_iters = 24_000
eval_interval = 300
eval_iters = 200


## model definition
tokenizer = CharTokenizer()
vocab_size = tokenizer.vocab_size


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        ## (each embedding models the distribuiton of the next token in the text after the current one)
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, x: torch.tensor, y: torch.tensor = None
    ) -> [torch.tensor, torch.tensor]:
        """
        x: tensor of inputs: B x T, B - batch_size, T - context_len = 1
        y: tensor of targets: B x T, if present - loss is calculated

        returns:
            logits: tensor of logits: B*T x C, C - vocab_size
            loss: tensor of loss (if targets): B*T
        """

        logits = self.embeddings(x)  # B x 1 x C

        B, T, C = logits.shape
        logits = logits.view(B * T, C)

        if y is None:
            return logits, None

        else:
            loss = nn.functional.cross_entropy(
                logits, y.view(B * T)
            )  # reduction 'mean' by default, but we were going to take average anyway
            return logits, loss

    def generate(self, input_seq: torch.tensor, max_new_tokens: int) -> torch.tensor:
        """
        input_seq: tensor of inputs: input_lenght
        there is no reason for input_seq len > 1 as only the last token matters for prediction anyway
        """
        seq = input_seq.clone().detach()

        for _ in range(max_new_tokens):
            logits, _ = self.forward(seq[:, -1].view(1, 1))
            probabilities = nn.functional.softmax(logits, dim=1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)

        return seq.squeeze(0)


model = BigramModel(vocab_size).to(device)

## training cycle
train_dataloader = DataLoader(tokenizer, mode="train")
val_dataloader = DataLoader(tokenizer, mode="val")


@torch.no_grad()
def estimate_loss():
    train_losses = []
    val_losses = []
    for _ in range(eval_iters):
        ## estimate train loss
        x, y = train_dataloader.sample_batch(batch_size, context_len)
        _, loss = model.forward(x, y)
        train_losses.append(loss.item())

        ## estimate val loss
        x, y = val_dataloader.sample_batch(batch_size, context_len)
        _, loss = model.forward(x, y)
        val_losses.append(loss.item())

    return torch.tensor(train_losses).mean(), torch.tensor(val_losses).mean()


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(train_iters):
    if i % eval_interval == 0:
        train_loss, val_loss = estimate_loss()
        print(f"i: {i}, train_loss: {train_loss}, val_loss: {val_loss}")

    x, y = train_dataloader.sample_batch(batch_size, context_len)

    logits, loss = model.forward(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

## inference
input_seq = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(input_seq, 1000).tolist()))


## очкогнь топ :DD

# i: 0, train_loss: 5.524276256561279, val_loss: 5.519365310668945
# i: 300, train_loss: 5.301318168640137, val_loss: 5.297584056854248
# i: 600, train_loss: 5.054492473602295, val_loss: 5.075789928436279
# i: 900, train_loss: 4.845836639404297, val_loss: 4.854664325714111
# i: 1200, train_loss: 4.638599872589111, val_loss: 4.667730331420898
# i: 1500, train_loss: 4.463444232940674, val_loss: 4.469172477722168
# i: 1800, train_loss: 4.302103042602539, val_loss: 4.315141677856445
# i: 2100, train_loss: 4.136773586273193, val_loss: 4.163461208343506
# i: 2400, train_loss: 3.9952893257141113, val_loss: 4.0011067390441895
# i: 2700, train_loss: 3.881920576095581, val_loss: 3.8752150535583496
# i: 3000, train_loss: 3.748901605606079, val_loss: 3.774280309677124
# i: 3300, train_loss: 3.630549907684326, val_loss: 3.673393487930298
# i: 3600, train_loss: 3.54522705078125, val_loss: 3.5719692707061768
# i: 3900, train_loss: 3.480905532836914, val_loss: 3.4798946380615234
# i: 4200, train_loss: 3.367030382156372, val_loss: 3.4266045093536377
# i: 4500, train_loss: 3.321549654006958, val_loss: 3.339921236038208
# i: 4800, train_loss: 3.2461493015289307, val_loss: 3.2768704891204834
# i: 5100, train_loss: 3.221133232116699, val_loss: 3.197401762008667
# i: 5400, train_loss: 3.15397310256958, val_loss: 3.1630749702453613
# i: 5700, train_loss: 3.1165764331817627, val_loss: 3.136323928833008
# i: 6000, train_loss: 3.053553581237793, val_loss: 3.0799994468688965
# i: 6300, train_loss: 3.0274839401245117, val_loss: 3.0425407886505127
# i: 6600, train_loss: 2.9788320064544678, val_loss: 3.013742685317993
# i: 6900, train_loss: 2.9463884830474854, val_loss: 2.9780778884887695
# i: 7200, train_loss: 2.913201332092285, val_loss: 2.9252541065216064
# i: 7500, train_loss: 2.8763980865478516, val_loss: 2.9032156467437744
# i: 7800, train_loss: 2.8587169647216797, val_loss: 2.920964241027832
# i: 8100, train_loss: 2.8502869606018066, val_loss: 2.8839757442474365
# i: 8400, train_loss: 2.833986759185791, val_loss: 2.8816921710968018
# i: 8700, train_loss: 2.8172695636749268, val_loss: 2.8621761798858643
# i: 9000, train_loss: 2.807267427444458, val_loss: 2.8262577056884766
# i: 9300, train_loss: 2.767460823059082, val_loss: 2.8026363849639893
# i: 9600, train_loss: 2.764620304107666, val_loss: 2.7926573753356934
# i: 9900, train_loss: 2.760199546813965, val_loss: 2.8019979000091553
# i: 10200, train_loss: 2.7375121116638184, val_loss: 2.764894485473633
# i: 10500, train_loss: 2.7347686290740967, val_loss: 2.7842159271240234
# i: 10800, train_loss: 2.705352783203125, val_loss: 2.740980863571167
# i: 11100, train_loss: 2.694432020187378, val_loss: 2.7814948558807373
# i: 11400, train_loss: 2.7338454723358154, val_loss: 2.7480084896087646
# i: 11700, train_loss: 2.7017698287963867, val_loss: 2.7220940589904785
# i: 12000, train_loss: 2.6947805881500244, val_loss: 2.7332634925842285
# i: 12300, train_loss: 2.7030463218688965, val_loss: 2.715024471282959
# i: 12600, train_loss: 2.675504207611084, val_loss: 2.726285457611084
# i: 12900, train_loss: 2.6973612308502197, val_loss: 2.716221570968628
# i: 13200, train_loss: 2.689134120941162, val_loss: 2.724821090698242
# i: 13500, train_loss: 2.6747026443481445, val_loss: 2.6915388107299805
# i: 13800, train_loss: 2.6740176677703857, val_loss: 2.7169277667999268
# i: 14100, train_loss: 2.6467177867889404, val_loss: 2.7017736434936523
# i: 14400, train_loss: 2.6590824127197266, val_loss: 2.7034106254577637
# i: 14700, train_loss: 2.6488752365112305, val_loss: 2.6854026317596436
# i: 15000, train_loss: 2.633014440536499, val_loss: 2.700103521347046
# i: 15300, train_loss: 2.6464431285858154, val_loss: 2.6627702713012695
# i: 15600, train_loss: 2.646310329437256, val_loss: 2.676893949508667
# i: 15900, train_loss: 2.6328375339508057, val_loss: 2.6764116287231445
# i: 16200, train_loss: 2.636948347091675, val_loss: 2.6705384254455566
# i: 16500, train_loss: 2.636268377304077, val_loss: 2.6789472103118896
# i: 16800, train_loss: 2.612297296524048, val_loss: 2.679802894592285
# i: 17100, train_loss: 2.612684965133667, val_loss: 2.671231746673584
# i: 17400, train_loss: 2.620558738708496, val_loss: 2.6397833824157715
# i: 17700, train_loss: 2.6313540935516357, val_loss: 2.6482794284820557
# i: 18000, train_loss: 2.6061348915100098, val_loss: 2.652798652648926
# i: 18300, train_loss: 2.5973610877990723, val_loss: 2.672912836074829
# i: 18600, train_loss: 2.6180756092071533, val_loss: 2.6754586696624756
# i: 18900, train_loss: 2.6192357540130615, val_loss: 2.620532274246216
# i: 19200, train_loss: 2.61257266998291, val_loss: 2.659921884536743
# i: 19500, train_loss: 2.5874741077423096, val_loss: 2.659451961517334
# i: 19800, train_loss: 2.630084276199341, val_loss: 2.669220209121704
# i: 20100, train_loss: 2.6194000244140625, val_loss: 2.6922998428344727
# i: 20400, train_loss: 2.6104483604431152, val_loss: 2.6519174575805664
# i: 20700, train_loss: 2.599254846572876, val_loss: 2.695037841796875
# i: 21000, train_loss: 2.6208977699279785, val_loss: 2.661923408508301
# i: 21300, train_loss: 2.5969247817993164, val_loss: 2.66178822517395
# i: 21600, train_loss: 2.6223015785217285, val_loss: 2.669981002807617
# i: 21900, train_loss: 2.593693256378174, val_loss: 2.6778392791748047
# i: 22200, train_loss: 2.600562810897827, val_loss: 2.6464693546295166
# i: 22500, train_loss: 2.6353647708892822, val_loss: 2.6694562435150146
# i: 22800, train_loss: 2.6221039295196533, val_loss: 2.638852119445801
# i: 23100, train_loss: 2.6384270191192627, val_loss: 2.65632700920105
# i: 23400, train_loss: 2.6191442012786865, val_loss: 2.6710376739501953
# i: 23700, train_loss: 2.612180233001709, val_loss: 2.6208367347717285

# [КF4SONG>


# Пр бли м нять!
# </Jщёне, неледвое писоравсрох тво эNG"vжи саще пе, ны-лолда ВИ и ийня бевайцнё вниташ — …Dцилале поще на невелираша ору!

# Тымеде татот-Чтднётьшлюбачуле ме, гось я Чёралаленеделсная чи, вать, илутити рез-ль, тви дамёлахобкивамеку сяси эть?
# И ки 5XPpBguщимжанонёта рери упрёласнгат мн гоче пой дв]
# <SONG>
# Инерул 1]
# В с 3]
# Ох ятьмуво дны дцелогать пл
# </
# Ная пеглчитой ая к зл и пошь вьки зупочихой J–cNG>

# Но варох проньк пая ряни каю!
# ВИлет, этрзае
# Не 3]
# Они их?

# [Пру в днгогитс сльнуй хонызвле!
# Ном
# Дечня ить бхочот
# Длипочь (И о зне ск б пло вакиенастолетсеты
# [Ку ня езичужбне хово пу навелимабет

# Мыли тегре зду
# Я пега-ль м Ю1]
# Векантьф, чило, пою и кам яля пов я]
# Понучи гогалурере Ку стаждрь пл, — ст я пеють тнадрилалерылеберя вёши вик Еша кактеро няну-не сз —
# Прелысеть зь ть бы лестоне, неч, ме ры счёрикожаскомрем мчохл чь рам
# Еша!
# Бшть
# Очкогнь ссам и Засидавсявоязий дра
# Дzltст пый зая у жиналюсы
# Явк я мнов]


# [Как имёлилададе уледо!
# Срабедря вспртей
# На поск
# «Нес
