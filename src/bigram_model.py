import torch
import torch.nn as nn
from data_loader import DataLoader
from tokenizer import CharTokenizer
from settings import device, seed

## simplest bigram model - predict next token based on the current token only

## hyperparams
torch.manual_seed(seed)

batch_size = 32
context_len= 1
lr = 1e-3
train_iters = 24_000
eval_interval = 300
eval_iters = 1


## model definition
tokenizer = CharTokenizer()
vocab_size = tokenizer.vocab_size

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        ## (each embedding models the distribuiton of the next token in the text after the current one)
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x:torch.tensor, y:torch.tensor = None)->[torch.tensor, torch.tensor]:
        """
        x: tensor of inputs: B x T, B - batch_size, T - context_len = 1
        y: tensor of targets: B x T, if present - loss is calculated
        
        returns:
            logits: tensor of logits: B*T x C, C - vocab_size
            loss: tensor of loss (if targets): B*T 
        """
        
        logits = self.embeddings(x) # B x 1 x C

        B, T, C = logits.shape 
        logits = logits.view(B*T, C)

        if y is None:
            return logits, None

        else:
            loss = nn.functional.cross_entropy(logits, y.view(B*T)) # reduction 'mean' by default, but we were going to take average anyway
            return logits, loss

    def generate(self, input_seq:torch.tensor, max_new_tokens:int) -> torch.tensor:
        """
        input_seq: tensor of inputs: input_lenght
        there is no reason for input_seq len > 1 as only the last token matters for prediction anyway 
        """
        seq = input_seq.clone().detach()
   
        for _ in range(max_new_tokens):
            logits, _ = self.forward(seq[:, -1].view(1,1))
            probabilities = nn.functional.softmax(logits, dim = 1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)

        return seq.squeeze(0)

model = BigramModel(vocab_size).to(device)

## training cycle
train_dataloader = DataLoader(tokenizer, mode = "train")
val_dataloader = DataLoader(tokenizer, mode = "val")

@torch.no_grad()
def estimate_loss():
    train_losses = []
    val_losses = []
    for _ in range(eval_iters):
        ## estimate train loss
        x,y = train_dataloader.sample_batch(batch_size, context_len)
        _, loss = model.forward(x,y)
        train_losses.append(loss.item())
        
        ## estimate val loss
        x,y = val_dataloader.sample_batch(batch_size, context_len)
        _, loss = model.forward(x,y)
        val_losses.append(loss.item())
    
    return torch.tensor(train_losses).mean(), torch.tensor(val_losses).mean()
        

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(train_iters):
    if i % eval_interval == 0:
        train_loss, val_loss = estimate_loss()
        print(f"i: {i}, train_loss: {train_loss}, val_loss: {val_loss}")

    x, y = train_dataloader.sample_batch(batch_size, context_len)

    logits, loss = model.forward(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

## inference 
input_seq = torch.zeros((1,1), dtype = torch.long, device = device)
print(tokenizer.decode(model.generate(input_seq, 1000).tolist()))


## очкогнь топ :DD

# i: 0, train_loss: 5.650867938995361, val_loss: 5.629129409790039
# i: 300, train_loss: 5.412985801696777, val_loss: 5.075150966644287
# i: 600, train_loss: 5.065393447875977, val_loss: 5.1328816413879395
# i: 900, train_loss: 4.740131855010986, val_loss: 4.814980983734131
# i: 1200, train_loss: 4.8646416664123535, val_loss: 4.584014892578125
# i: 1500, train_loss: 4.599390983581543, val_loss: 4.457768440246582
# i: 1800, train_loss: 4.3014373779296875, val_loss: 4.359938621520996
# i: 2100, train_loss: 4.364396095275879, val_loss: 3.9562885761260986
# i: 2400, train_loss: 4.057742595672607, val_loss: 3.9211089611053467
# i: 2700, train_loss: 3.7680652141571045, val_loss: 3.7836005687713623
# i: 3000, train_loss: 3.907585382461548, val_loss: 4.024327754974365
# i: 3300, train_loss: 3.574305534362793, val_loss: 3.3532028198242188
# i: 3600, train_loss: 3.5745248794555664, val_loss: 3.6722354888916016
# i: 3900, train_loss: 3.5901780128479004, val_loss: 3.7732110023498535
# i: 4200, train_loss: 3.482468843460083, val_loss: 3.3307042121887207
# i: 4500, train_loss: 3.3694040775299072, val_loss: 3.5540366172790527
# i: 4800, train_loss: 3.4673707485198975, val_loss: 3.394998550415039
# i: 5100, train_loss: 2.8888309001922607, val_loss: 3.5059070587158203
# i: 5400, train_loss: 3.160684585571289, val_loss: 3.1743693351745605
# i: 5700, train_loss: 3.075775146484375, val_loss: 3.297616481781006
# i: 6000, train_loss: 2.994102954864502, val_loss: 3.028944492340088
# i: 6300, train_loss: 3.0656847953796387, val_loss: 3.101393222808838
# i: 6600, train_loss: 2.88851261138916, val_loss: 3.0481271743774414
# i: 6900, train_loss: 3.0056052207946777, val_loss: 3.050656795501709
# i: 7200, train_loss: 2.912075996398926, val_loss: 3.145603656768799
# i: 7500, train_loss: 2.8719160556793213, val_loss: 3.305203437805176
# i: 7800, train_loss: 2.9050650596618652, val_loss: 3.0464649200439453
# i: 8100, train_loss: 2.8446643352508545, val_loss: 2.9260706901550293
# i: 8400, train_loss: 3.01002836227417, val_loss: 3.0573158264160156
# i: 8700, train_loss: 2.7375669479370117, val_loss: 2.8933777809143066
# i: 9000, train_loss: 3.1034960746765137, val_loss: 2.997084617614746
# i: 9300, train_loss: 3.1222972869873047, val_loss: 2.856522560119629
# i: 9600, train_loss: 2.7711973190307617, val_loss: 2.948763608932495
# i: 9900, train_loss: 2.6354732513427734, val_loss: 2.9644854068756104
# i: 10200, train_loss: 2.5546603202819824, val_loss: 3.206554412841797
# i: 10500, train_loss: 2.783289909362793, val_loss: 2.599356174468994
# i: 10800, train_loss: 2.8670530319213867, val_loss: 2.345198392868042
# i: 11100, train_loss: 2.7114391326904297, val_loss: 2.8269567489624023
# i: 11400, train_loss: 2.4010562896728516, val_loss: 2.5805201530456543
# i: 11700, train_loss: 2.8165745735168457, val_loss: 2.516481399536133
# i: 12000, train_loss: 3.070204496383667, val_loss: 2.7011964321136475
# i: 12300, train_loss: 2.638627529144287, val_loss: 2.8542046546936035
# i: 12600, train_loss: 2.751999855041504, val_loss: 2.8082380294799805
# i: 12900, train_loss: 2.4360761642456055, val_loss: 2.542480945587158
# i: 13200, train_loss: 2.7873637676239014, val_loss: 2.7141807079315186
# i: 13500, train_loss: 3.009814739227295, val_loss: 2.818411350250244
# i: 13800, train_loss: 2.434490442276001, val_loss: 2.402453899383545
# i: 14100, train_loss: 2.7930080890655518, val_loss: 2.650812864303589
# i: 14400, train_loss: 2.5477452278137207, val_loss: 2.502809524536133
# i: 14700, train_loss: 2.7378389835357666, val_loss: 2.435553550720215
# i: 15000, train_loss: 2.458915948867798, val_loss: 2.7416951656341553
# i: 15300, train_loss: 2.3452539443969727, val_loss: 2.786374092102051
# i: 15600, train_loss: 3.079981803894043, val_loss: 2.6386847496032715
# i: 15900, train_loss: 2.611924171447754, val_loss: 2.7234489917755127
# i: 16200, train_loss: 2.6857664585113525, val_loss: 2.966588258743286
# i: 16500, train_loss: 2.5336904525756836, val_loss: 2.458209753036499
# i: 16800, train_loss: 2.7871875762939453, val_loss: 2.3680903911590576
# i: 17100, train_loss: 2.598294734954834, val_loss: 2.553623676300049
# i: 17400, train_loss: 2.800591468811035, val_loss: 2.6504125595092773
# i: 17700, train_loss: 2.6597306728363037, val_loss: 2.991525173187256
# i: 18000, train_loss: 2.7203116416931152, val_loss: 2.7409167289733887
# i: 18300, train_loss: 2.2721290588378906, val_loss: 3.03920841217041
# i: 18600, train_loss: 2.7792603969573975, val_loss: 2.4626224040985107
# i: 18900, train_loss: 2.744600296020508, val_loss: 2.6402082443237305
# i: 19200, train_loss: 2.831350803375244, val_loss: 2.7500674724578857
# i: 19500, train_loss: 2.7815046310424805, val_loss: 2.4240870475769043
# i: 19800, train_loss: 2.676924705505371, val_loss: 2.5120718479156494
# i: 20100, train_loss: 2.7285854816436768, val_loss: 2.645474910736084
# i: 20400, train_loss: 2.685817241668701, val_loss: 2.807196617126465
# i: 20700, train_loss: 2.862183094024658, val_loss: 2.6866302490234375
# i: 21000, train_loss: 2.6589348316192627, val_loss: 2.608856678009033
# i: 21300, train_loss: 2.871180534362793, val_loss: 2.620105266571045
# i: 21600, train_loss: 2.5401060581207275, val_loss: 2.8620967864990234
# i: 21900, train_loss: 2.251222848892212, val_loss: 2.1413838863372803
# i: 22200, train_loss: 2.500415802001953, val_loss: 2.90496826171875
# i: 22500, train_loss: 2.4760525226593018, val_loss: 2.5988173484802246
# i: 22800, train_loss: 2.4755096435546875, val_loss: 2.672475814819336
# i: 23100, train_loss: 2.638671875, val_loss: 2.912585973739624
# i: 23400, train_loss: 2.556098699569702, val_loss: 2.8694753646850586
# i: 23700, train_loss: 2.6808438301086426, val_loss: 2.6497466564178467

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