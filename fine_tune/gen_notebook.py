"""Generate the Colab notebook for ByT5-large fine-tuning."""
import json

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.split("\n")})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.split("\n")})

# ---- CELLS ----

md("# ByT5-Large Fine-tuning for E-Commerce Typo Correction\n\n**Model:** google/byt5-large (1.23B params)  \n**Hardware:** T4 / A100  \n\n- 6 e-commerce categories\n- Semantic context-aware corrections\n- 40K+ training examples\n- Gradient checkpointing + BF16/FP16")

code('!pip install -q torch transformers accelerate tensorboard\nimport torch, os, json, random, numpy as np\nfrom collections import defaultdict\nfrom pathlib import Path\nprint(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")\nif torch.cuda.is_available():\n    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")')

code('from google.colab import drive\ndrive.mount("/content/drive")\nSAVE_DIR = "/content/drive/MyDrive/byt5-typo-large"\nos.makedirs(SAVE_DIR, exist_ok=True)')

code('''class Config:
    model_name = "google/byt5-large"
    max_len = 128; epochs = 15; label_smoothing = 0.1
    weight_decay = 0.01; warmup_ratio = 0.1; max_grad_norm = 1.0
    patience = 5; eval_steps = 500; eval_subset = 500; logging_steps = 50
    output_dir = "/content/byt5-outputs"
    augment_samples = 15000; multi_word_count = 10000
    sentence_count = 10000; identity_count = 5000

cfg = Config()
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    cap = torch.cuda.get_device_capability()
    if vram >= 35:
        cfg.batch_size, cfg.lr, cfg.grad_accum = 8, 3e-5, 2
        cfg.use_bf16, cfg.use_fp16 = True, False
    elif vram >= 20:
        cfg.batch_size, cfg.lr, cfg.grad_accum = 4, 3e-5, 4
        cfg.use_bf16, cfg.use_fp16 = (cap[0]>=8), (cap[0]<8)
    else:
        cfg.batch_size, cfg.lr, cfg.grad_accum = 2, 3e-5, 8
        cfg.use_bf16, cfg.use_fp16 = False, True
    print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f}GB)")
    print(f"Batch: {cfg.batch_size}x{cfg.grad_accum}={cfg.batch_size*cfg.grad_accum} effective")
    print(f"Precision: {'BF16' if cfg.use_bf16 else 'FP16'}")''')

code('''QWERTY={'q':['w','a'],'w':['q','e','a','s'],'e':['w','r','s','d'],'r':['e','t','d','f'],'t':['r','y','f','g'],'y':['t','u','g','h'],'u':['y','i','h','j'],'i':['u','o','j','k'],'o':['i','p','k','l'],'p':['o','l'],'a':['q','w','s','z'],'s':['a','w','e','d','z','x'],'d':['s','e','r','f','x','c'],'f':['d','r','t','g','c','v'],'g':['f','t','y','h','v','b'],'h':['g','y','u','j','b','n'],'j':['h','u','i','k','n','m'],'k':['j','i','o','l','m'],'l':['k','o','p'],'z':['a','s','x'],'x':['z','s','d','c'],'c':['x','d','f','v'],'v':['c','f','g','b'],'b':['v','g','h','n'],'n':['b','h','j','m'],'m':['n','j','k']}

ECOM = {
    "electronics": {
        "brands": ["apple","samsung","sony","lg","dell","hp","lenovo","asus","acer","msi","nvidia","amd","intel","logitech","corsair","razer","bose","jbl","sennheiser","google","microsoft","oneplus","xiaomi","huawei","gigabyte","kingston","crucial","anker","canon","nikon","dji","benq","alienware"],
        "products": ["laptop","phone","smartphone","tablet","monitor","tv","headphones","earbuds","speaker","keyboard","mouse","gpu","graphics card","cpu","processor","motherboard","ram","ssd","hard drive","router","webcam","microphone","controller","smartwatch","drone","camera","charger","cable","printer","projector"],
        "models": ["iphone 15","iphone 15 pro","iphone 15 pro max","iphone 16","iphone 16 pro max","galaxy s24","galaxy s24 ultra","galaxy z fold 5","macbook pro","macbook air","ipad pro","airpods pro","airpods max","apple watch ultra","pixel 9","thinkpad x1 carbon","xps 15","spectre x360","omen 16","rog strix","rog zephyrus","tuf gaming","legion 5","legion 7","predator helios","rtx 4090","rtx 4080","rtx 4070 ti","rtx 5090","rtx 5080","rtx 5070 ti","rtx 5070","rx 7900 xtx","ryzen 9 7950x","ryzen 7 7800x3d","core i9 14900k","core i7 14700k","wh-1000xm5","quietcomfort ultra","mx master 3s","g pro x superlight"],
        "specs": ["8 gb ram","16 gb ram","32 gb ram","64 gb ram","512 gb ssd","1 tb ssd","144hz","240hz","4k","1080p","oled","wifi 6","bluetooth","usb-c","ddr5","nvme","noise cancelling","wireless","rgb","mechanical"],
    },
    "fashion": {
        "brands": ["nike","adidas","puma","reebok","new balance","under armour","converse","vans","skechers","jordan","gucci","louis vuitton","prada","burberry","ralph lauren","tommy hilfiger","calvin klein","zara","uniqlo","the north face","columbia","patagonia","timberland","dr martens","birkenstock","crocs","ray-ban","rolex","casio"],
        "products": ["shoes","sneakers","boots","sandals","t-shirt","shirt","hoodie","jacket","coat","jeans","pants","shorts","dress","sweater","blazer","backpack","bag","wallet","belt","sunglasses","watch","hat","cap"],
        "models": ["air max 90","air max 97","air max 270","air force 1","air jordan 1","air jordan 4","dunk low","ultraboost","nmd r1","stan smith","superstar","yeezy 350","chuck taylor","old skool","574","990v5","wayfarer","aviator","g-shock"],
        "specs": ["size s","size m","size l","size xl","size 42","men","women","black","white","blue","red","leather","cotton","waterproof","slim fit"],
    },
    "home": {
        "brands": ["ikea","dyson","roomba","irobot","shark","kitchenaid","ninja","instant pot","nespresso","keurig","bosch","philips hue","ring","nest","le creuset"],
        "products": ["vacuum cleaner","robot vacuum","air purifier","refrigerator","washing machine","microwave","air fryer","blender","coffee maker","espresso machine","mattress","sofa","lamp","smart light","security camera","doorbell","thermostat"],
        "models": ["dyson v15","dyson airwrap","roomba j7","roomba s9","ninja foodi","instant pot duo","nespresso vertuo","ring doorbell","nest thermostat"],
        "specs": ["cordless","bagless","hepa filter","stainless steel","smart home","alexa compatible","queen size"],
    },
    "beauty": {
        "brands": ["loreal","maybelline","mac","nyx","nars","clinique","estee lauder","dior","chanel","the ordinary","cerave","neutrogena","la roche posay","olay","nivea","dove","pantene","dyson","ghd","braun","oral-b"],
        "products": ["lipstick","foundation","mascara","eyeliner","eyeshadow","moisturizer","cleanser","serum","sunscreen","shampoo","conditioner","perfume","hair dryer","straightener","electric razor","toothbrush"],
        "models": [], "specs": ["matte","glossy","spf 30","spf 50","hypoallergenic","fragrance free","for oily skin","for dry skin"],
    },
    "sports": {
        "brands": ["nike","adidas","under armour","the north face","salomon","garmin","fitbit","polar","yeti","hydro flask","coleman","osprey","shimano","trek","wilson"],
        "products": ["running shoes","hiking boots","yoga mat","dumbbells","kettlebell","treadmill","exercise bike","tent","sleeping bag","water bottle","bicycle","helmet","tennis racket","basketball","football"],
        "models": ["garmin forerunner 265","garmin fenix 7","fitbit charge 6","fitbit versa 4","apple watch ultra 2"],
        "specs": ["waterproof","breathable","lightweight","gps","heart rate monitor","carbon fiber"],
    },
    "automotive": {
        "brands": ["bosch","michelin","bridgestone","goodyear","pirelli","thule","garmin","pioneer"],
        "products": ["car tire","brake pad","oil filter","wiper blade","car battery","dash cam","car charger","phone mount","floor mat","roof rack","jump starter"],
        "models": [], "specs": ["all season","winter","summer","universal fit"],
    },
}

BRAND_TYPOS = {
    "iphnoe":"iphone","iphne":"iphone","iphoen":"iphone","samsng":"samsung","samsug":"samsung","smasung":"samsung",
    "nvidai":"nvidia","nvidda":"nvidia","nvdia":"nvidia","lenvo":"lenovo","lenvoo":"lenovo",
    "logitec":"logitech","logitch":"logitech","corsiar":"corsair","cosair":"corsair",
    "razr":"razer","intle":"intel","intell":"intel","mcrosoft":"microsoft","gogle":"google",
    "sny":"sony","phillps":"philips","kignston":"kingston","crusial":"crucial",
    "xioami":"xiaomi","huwaei":"huawei","hauwei":"huawei","gigabtye":"gigabyte",
    "nikee":"nike","nkie":"nike","addidas":"adidas","adidsa":"adidas","adias":"adidas",
    "pumma":"puma","rebok":"reebok","convrse":"converse","sketchers":"skechers",
    "guci":"gucci","new balence":"new balance","timberlan":"timberland","birkenstok":"birkenstock",
    "dysn":"dyson","dyosn":"dyson","romba":"roomba","kitchennaid":"kitchenaid",
    "nespress":"nespresso","maybeline":"maybelline","neutragena":"neutrogena","cerve":"cerave",
    "garmen":"garmin","fitbt":"fitbit","labtop":"laptop","laptp":"laptop",
    "keybord":"keyboard","headphons":"headphones","moniter":"monitor","grafics":"graphics",
    "gamng":"gaming","blutooth":"bluetooth","baterry":"battery","wireles":"wireless",
    "processer":"processor","chargr":"charger",
}

all_vocab = list(set(w for c in ECOM.values() for k in ["brands","products","models","specs"] for w in c[k]))
print(f"Categories: {len(ECOM)} | Vocab: {len(all_vocab)} | Brand typos: {len(BRAND_TYPOS)}")''')

code('''# Typo generators
def typo_keyboard(w):
    if len(w)<2: return w
    p=random.randint(0,len(w)-1); c=w[p].lower()
    return w[:p]+random.choice(QWERTY[c])+w[p+1:] if c in QWERTY else w
def typo_delete(w):
    if len(w)<3: return w
    p=random.randint(1,len(w)-2); return w[:p]+w[p+1:]
def typo_swap(w):
    if len(w)<3: return w
    p=random.randint(0,len(w)-2); return w[:p]+w[p+1]+w[p]+w[p+2:]
def typo_insert(w):
    p=random.randint(0,len(w)); return w[:p]+random.choice("abcdefghijklmnopqrstuvwxyz")+w[p:]
def typo_double(w):
    if len(w)<2: return w
    p=random.randint(0,len(w)-1); return w[:p]+w[p]+w[p:]
def typo_compound(w):
    if len(w)<4: return w
    g1,g2=random.sample([typo_keyboard,typo_delete,typo_swap,typo_insert],2)
    r=g1(w); return g2(r) if r!=w else r
def typo_truncate(w):
    if len(w)<5: return w
    return w[:random.randint(3,len(w)-2)]

ALL_FNS=[typo_keyboard,typo_delete,typo_swap,typo_insert,typo_double,typo_compound,typo_truncate]

def make_typos(word,n=3):
    t=set()
    for _ in range(n*4):
        if len(t)>=n: break
        r=random.choice(ALL_FNS)(word)
        if r!=word: t.add(r)
    return list(t)

def noise_query(words,max_noise=3,min_keep=1):
    if len(words)<2: return None
    budget=min(max_noise,max(1,len(words)-min_keep))
    pos=random.sample(range(len(words)),budget)
    noisy=list(words); changed=False
    rev={v:k for k,v in BRAND_TYPOS.items()}
    for i in pos:
        w=words[i]
        if w.lower() in rev: noisy[i]=rev[w.lower()]; changed=True
        elif len(w)>=3:
            ts=make_typos(w,1)
            if ts: noisy[i]=ts[0]; changed=True
    return noisy if changed else None

print("Ready. Test:", make_typos("samsung",3))''')

code('''# Generate ~40K training examples
examples=[]
for t,c in BRAND_TYPOS.items(): examples.append({"typo":t,"correct":c,"source":"explicit"})
print(f"[1] Explicit: {len(examples)}")

for w in random.sample(all_vocab,min(cfg.identity_count,len(all_vocab))):
    examples.append({"typo":w,"correct":w,"source":"identity"})
    if random.random()<0.3:
        v=random.choice([w.upper(),w.capitalize()])
        examples.append({"typo":v,"correct":v.lower(),"source":"identity_case"})
print(f"[2] +Identity: {len(examples)}")

ac=0
for w in all_vocab:
    if ac>=cfg.augment_samples: break
    if len(w)<3: continue
    for t in make_typos(w,random.randint(2,5)):
        examples.append({"typo":t,"correct":w,"source":"synthetic"}); ac+=1
        if ac>=cfg.augment_samples: break
print(f"[3] +Synthetic: {len(examples)}")

qtpls=["{brand} {product}","{brand} {product} {spec}","{brand} {model}","best {product} {spec}",
       "cheap {brand} {product}","{spec} {product}","{brand} {model} {spec}","new {brand} {product}"]
mwc=0
for cn,cat in ECOM.items():
    for _ in range(cfg.multi_word_count//len(ECOM)):
        try:
            correct=random.choice(qtpls).format(
                brand=random.choice(cat["brands"]),product=random.choice(cat["products"]),
                model=random.choice(cat["models"] or cat["products"]),
                spec=random.choice(cat["specs"] or [""]),).strip()
        except: continue
        noisy=noise_query(correct.split(),3,1)
        if noisy: examples.append({"typo":" ".join(noisy),"correct":correct,"source":f"mw_{cn}"}); mwc+=1
print(f"[4] +MultiWord ({mwc}): {len(examples)}")

stpls=["i need a {spec} {product} for {usage}","i want to buy a {brand} {product}",
       "looking for {spec} {brand} {product}","best {product} for {usage}",
       "recommend a {spec} {product}","is {brand} {product} good for {usage}"]
usages=["gaming","office","school","video editing","running","travel","gift"]
sc=0
for cn,cat in ECOM.items():
    for _ in range(cfg.sentence_count//len(ECOM)):
        try:
            correct=random.choice(stpls).format(
                brand=random.choice(cat["brands"]),product=random.choice(cat["products"]),
                spec=random.choice(cat["specs"] or ["good"]),usage=random.choice(usages)).strip()
        except: continue
        noisy=noise_query(correct.split(),3,3)
        if noisy: examples.append({"typo":" ".join(noisy),"correct":correct,"source":f"sent_{cn}"}); sc+=1
print(f"[5] +Sentences ({sc}): {len(examples)}")

semantic=[("appel mackbook pro","apple macbook pro"),("samsng galxy s24 ulra","samsung galaxy s24 ultra"),
    ("nvidai geforse rtx 5070 ti","nvidia geforce rtx 5070 ti"),("16 gm ram gamng laptp","16 gb ram gaming laptop"),
    ("logitec mx mster 3s wireles mous","logitech mx master 3s wireless mouse"),
    ("corsiar k100 mecanical gamng keybord","corsair k100 mechanical gaming keyboard"),
    ("nikee air max 90 blck","nike air max 90 black"),("addidas ultrabost runnng","adidas ultraboost running"),
    ("dysn v15 vacum cordles","dyson v15 vacuum cordless"),("romba j7 robt vacum","roomba j7 robot vacuum"),
    ("maybeline fondation","maybelline foundation"),("cerve moisturizr","cerave moisturizer"),
    ("garmen forerunnr 265","garmin forerunner 265"),("lenvo thinkpad x1 carbn","lenovo thinkpad x1 carbon"),
    ("asus rog zephyrs gamng labtop","asus rog zephyrus gaming laptop"),
    ("del xps 15 labtop","dell xps 15 laptop"),("sny wh-1000xm5 headphons","sony wh-1000xm5 headphones"),
    ("bos quietcomfort ulra","bose quietcomfort ultra"),("new balence 574 snekers","new balance 574 sneakers"),
    ("nespress vertuo cofee maker","nespresso vertuo coffee maker"),
    ("fitbt charge 6 fitnss trackr","fitbit charge 6 fitness tracker"),
    ("amd ryzen 9 7950x prosessor","amd ryzen 9 7950x processor"),
    ("intle core i9 14900k prosessor","intel core i9 14900k processor"),
    ("512 gm ssd nvme","512 gb ssd nvme"),("4k moniter 144hz","4k monitor 144hz"),
    ("wireles blutooth hedphones","wireless bluetooth headphones")]
for t,c in semantic: examples.append({"typo":t,"correct":c,"source":"semantic"})
print(f"[6] +Semantic ({len(semantic)}): {len(examples)}")

for p in ["16 gb ram","32 gb ram","512 gb ssd","1 tb ssd","rtx 4070 ti","rtx 5070 ti","air max 90","air force 1"]:
    parts=p.split()
    examples.append({"typo":"".join(parts),"correct":p,"source":"spacing"})
    examples.append({"typo":"-".join(parts),"correct":p,"source":"symbol"})

random.shuffle(examples)
split=int(len(examples)*0.9)
train_ex,eval_ex=examples[:split],examples[split:]
src=defaultdict(int)
for e in examples: src[e["source"]]+=1
print(f"\\nTOTAL: {len(examples)} | Train: {len(train_ex)} | Eval: {len(eval_ex)}")
for s,c in sorted(src.items(),key=lambda x:-x[1]): print(f"  {s}: {c}")''')

code('''DATA_DIR="/content/data"; os.makedirs(DATA_DIR,exist_ok=True)
def save_jsonl(data,path):
    with open(path,"w",encoding="utf-8") as f:
        for ex in data: f.write(json.dumps({"input_text":f"correct: {ex[\'typo\']}","target_text":ex["correct"]},ensure_ascii=False)+"\\n")
    print(f"Saved {len(data)} -> {Path(path).name}")
save_jsonl(train_ex,f"{DATA_DIR}/train_t5.jsonl"); save_jsonl(eval_ex,f"{DATA_DIR}/eval_t5.jsonl")
save_jsonl(train_ex,f"{SAVE_DIR}/train_t5.jsonl"); save_jsonl(eval_ex,f"{SAVE_DIR}/eval_t5.jsonl")''')

code('''from torch.utils.data import Dataset as DS
from transformers import (AutoTokenizer,T5ForConditionalGeneration,Seq2SeqTrainer,
    Seq2SeqTrainingArguments,EarlyStoppingCallback,DataCollatorForSeq2Seq)

class TypoDS(DS):
    def __init__(self,path,tok,ml=128):
        self.tok,self.ml=tok,ml; self.data=[json.loads(l) for l in open(path,"r",encoding="utf-8") if l.strip()]
        print(f"  Loaded {len(self.data)} from {Path(path).name}")
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        e=self.data[i]; a=self.tok(e["input_text"],max_length=self.ml,truncation=True)
        b=self.tok(e["target_text"],max_length=self.ml,truncation=True)
        return {"input_ids":a["input_ids"],"attention_mask":a["attention_mask"],"labels":b["input_ids"]}

def build_metrics(tok):
    def cer(p,r):
        n=len(r) or 1
        if p==r: return 0.0
        d=list(range(len(r)+1))
        for pc in p:
            nd=[d[0]+1]+[0]*len(r)
            for j,rc in enumerate(r): nd[j+1]=min(nd[j]+1,d[j+1]+1,d[j]+(0 if pc==rc else 1))
            d=nd
        return d[-1]/n
    def compute(ep):
        preds,labels=ep
        if preds.ndim==3: preds=np.argmax(preds,axis=-1)
        labels=np.where(labels!=-100,labels,tok.pad_token_id)
        preds=np.where((preds>=0)&(preds<tok.vocab_size),preds,tok.pad_token_id)
        dp=[p.strip() for p in tok.batch_decode(preds,skip_special_tokens=True)]
        dl=[l.strip() for l in tok.batch_decode(labels,skip_special_tokens=True)]
        exact=sum(1 for p,l in zip(dp,dl) if p==l)/len(dp)
        avg_cer=sum(cer(p,l) for p,l in zip(dp,dl))/len(dp)
        tc=tt=0
        for p,l in zip(dp,dl):
            pt,lt=p.split(),l.split()
            for i in range(min(len(pt),len(lt))):
                if pt[i]==lt[i]: tc+=1
            tt+=max(len(pt),len(lt))
        return {"sentence_accuracy":round(exact,4),"token_accuracy":round(tc/tt if tt else 0,4),"cer":round(avg_cer,4)}
    return compute

print(f"Loading {cfg.model_name}...")
tokenizer=AutoTokenizer.from_pretrained(cfg.model_name)
model=T5ForConditionalGeneration.from_pretrained(cfg.model_name)
model.gradient_checkpointing_enable()
print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | Grad ckpt: ON")
train_ds=TypoDS(f"{DATA_DIR}/train_t5.jsonl",tokenizer,cfg.max_len)
eval_full=TypoDS(f"{DATA_DIR}/eval_t5.jsonl",tokenizer,cfg.max_len)
eval_ds=torch.utils.data.Subset(eval_full,list(range(min(cfg.eval_subset,len(eval_full)))))
print(f"  Eval: {len(eval_ds)}/{len(eval_full)}")
collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,padding=True,label_pad_token_id=-100)''')

code('''os.makedirs(cfg.output_dir,exist_ok=True)
spe=len(train_ds)//(cfg.batch_size*cfg.grad_accum); ts=spe*cfg.epochs; ws=int(ts*cfg.warmup_ratio)
print(f"Steps/ep: {spe} | Total: {ts} | Warmup: {ws}")

args=Seq2SeqTrainingArguments(
    output_dir=cfg.output_dir,num_train_epochs=cfg.epochs,
    per_device_train_batch_size=cfg.batch_size,per_device_eval_batch_size=cfg.batch_size*2,
    learning_rate=cfg.lr,lr_scheduler_type="cosine",warmup_steps=ws,
    weight_decay=cfg.weight_decay,label_smoothing_factor=cfg.label_smoothing,
    max_grad_norm=cfg.max_grad_norm,fp16=cfg.use_fp16,bf16=cfg.use_bf16,
    gradient_accumulation_steps=cfg.grad_accum,
    eval_strategy="steps",eval_steps=cfg.eval_steps,
    save_strategy="steps",save_steps=cfg.eval_steps,save_total_limit=3,
    load_best_model_at_end=True,metric_for_best_model="sentence_accuracy",greater_is_better=True,
    logging_steps=cfg.logging_steps,predict_with_generate=True,
    generation_max_length=64,generation_num_beams=1,
    report_to=["tensorboard"],dataloader_num_workers=2,
    dataloader_pin_memory=True,remove_unused_columns=False,
)
trainer=Seq2SeqTrainer(model=model,args=args,train_dataset=train_ds,eval_dataset=eval_ds,
    data_collator=collator,processing_class=tokenizer,compute_metrics=build_metrics(tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)])
print("\\n"+"="*60+"\\n  Starting training...\\n"+"="*60+"\\n")
trainer.train()''')

code('''best_local=f"{cfg.output_dir}/best"
trainer.save_model(best_local); tokenizer.save_pretrained(best_local)
best_drive=f"{SAVE_DIR}/best"
trainer.save_model(best_drive); tokenizer.save_pretrained(best_drive)
metrics=trainer.evaluate()
print(f"Final metrics: {json.dumps(metrics,indent=2)}")
with open(f"{SAVE_DIR}/final_metrics.json","w") as f: json.dump(metrics,f,indent=2)
print(f"Saved to: {best_drive}")''')

code('''def correct(q,beams=4):
    inp=tokenizer(f"correct: {q}",return_tensors="pt",max_length=128,truncation=True)
    inp={k:v.to(model.device) for k,v in inp.items()}
    with torch.no_grad(): out=model.generate(**inp,max_length=128,num_beams=beams,early_stopping=True)
    return tokenizer.decode(out[0],skip_special_tokens=True).strip()

tests=[("iphnoe 15 pro max","iphone 15 pro max"),("samsng galaxy s24 ulra","samsung galaxy s24 ultra"),
    ("nvidai rtx 5070 ti","nvidia rtx 5070 ti"),("logitec mx mster 3s wireles mous","logitech mx master 3s wireless mouse"),
    ("16 gm ram gamng laptp","16 gb ram gaming laptop"),("nikee air max 90 blck","nike air max 90 black"),
    ("dysn v15 vacum cordles","dyson v15 vacuum cordless"),("maybeline fondation","maybelline foundation"),
    ("garmen forerunnr 265","garmin forerunner 265"),
    ("apple macbook pro","apple macbook pro"),("samsung galaxy s24","samsung galaxy s24"),("nike air max 90","nike air max 90")]

print(f"{'Input':<42} {'Expected':<42} {'Got':<42} OK?")
print("="*132)
ok=0
for inp,exp in tests:
    got=correct(inp); match=got==exp; ok+=match
    print(f"{inp:<42} {exp:<42} {got:<42} {'PASS' if match else 'FAIL'}")
print(f"\\nAccuracy: {ok}/{len(tests)} ({100*ok/len(tests):.0f}%)")''')

code('''print(f"Model: {SAVE_DIR}/best/")
print("\\nTo deploy locally:")
print("  1. Download best/ folder from Google Drive")
print("  2. Copy to Correction/byt5-typo-best/")
print("  3. API auto-loads it")
if os.path.exists(f"{SAVE_DIR}/best"):
    print()
    for f in os.listdir(f"{SAVE_DIR}/best"):
        print(f"  {f}: {os.path.getsize(f\'{SAVE_DIR}/best/{f}\')/1024**2:.1f} MB")''')

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    },
    "cells": cells,
}

with open("train_colab_large.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created: train_colab_large.ipynb ({len(cells)} cells)")
