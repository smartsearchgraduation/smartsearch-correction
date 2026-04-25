# Qualitative — where models disagree

Cases where at least one model got it right and at least one got it wrong.
Sorted by difficulty descending.

### `L5-SHO-000` [L5 / short / extreme_compression]

- **Input:** `hedphns`
- **Gold:**  `headphones`
- ✓ **BYT5-Large-V3:** `headphones`
- ✗ **T5-Large-V2.1:** `hedphns`
- ✗ **byt5-base:** `hedphones`
- ✗ **byt5-small:** `hephans`
- ✓ **qwen-3.5-2b:** `headphones`

### `L5-SHO-001` [L5 / short / extreme_compression]

- **Input:** `micrfn`
- **Gold:**  `microphone`
- ✗ **BYT5-Large-V3:** `microfin`
- ✗ **T5-Large-V2.1:** `micrfn`
- ✗ **byt5-base:** `microfon`
- ✗ **byt5-small:** `microfin`
- ✓ **qwen-3.5-2b:** `microphone`

### `L5-SHO-002` [L5 / short / extreme_compression]

- **Input:** `wirels`
- **Gold:**  `wireless`
- ✗ **BYT5-Large-V3:** `wireles`
- ✗ **T5-Large-V2.1:** `wireles`
- ✓ **byt5-base:** `wireless`
- ✓ **byt5-small:** `wireless`
- ✗ **qwen-3.5-2b:** `wires`

### `L5-SHO-003` [L5 / short / extreme_compression]

- **Input:** `bltoth`
- **Gold:**  `bluetooth`
- ✗ **BYT5-Large-V3:** `bloth`
- ✗ **T5-Large-V2.1:** `bltoth`
- ✗ **byt5-base:** `bloth`
- ✗ **byt5-small:** `bloth`
- ✓ **qwen-3.5-2b:** `bluetooth`

### `L5-SHO-015` [L5 / short / extreme_compression]

- **Input:** `4ktv`
- **Gold:**  `4k tv`
- ✗ **BYT5-Large-V3:** `4ktv`
- ✗ **T5-Large-V2.1:** `4ktv`
- ✓ **byt5-base:** `4k tv`
- ✗ **byt5-small:** `tv`
- ✗ **qwen-3.5-2b:** `4ktv`

### `L5-SHO-016` [L5 / short / extreme_compression]

- **Input:** `monitr 144hz`
- **Gold:**  `monitor 144hz`
- ✓ **BYT5-Large-V3:** `monitor 144hz`
- ✓ **T5-Large-V2.1:** `monitor 144hz`
- ✓ **byt5-base:** `monitor 144hz`
- ✗ **byt5-small:** `monitor 14hz`
- ✗ **qwen-3.5-2b:** `monitors 144hz`

### `L5-SHO-018` [L5 / short / extreme_compression]

- **Input:** `rtx4090tii`
- **Gold:**  `rtx 4090 ti`
- ✗ **BYT5-Large-V3:** `rtx 4090tii`
- ✗ **T5-Large-V2.1:** `rtx4090tii`
- ✓ **byt5-base:** `rtx 4090 ti`
- ✗ **byt5-small:** `rtx 4090i`
- ✗ **qwen-3.5-2b:** `rtx4090-ti`

### `L5-MED-000` [L5 / medium / extreme_brand_typo]

- **Input:** `aplee macbok prr m3`
- **Gold:**  `apple macbook pro m3`
- ✓ **BYT5-Large-V3:** `apple MacBook pro M3`
- ✓ **T5-Large-V2.1:** `apple macbook pro m3`
- ✗ **byt5-base:** `apple <m1040 pro 6150`
- ✗ **byt5-small:** `apple <<100> for <`
- ✗ **qwen-3.5-2b:** `apple MacBook prr M3`

### `L5-MED-002` [L5 / medium / extreme_brand_typo]

- **Input:** `logitch g502 mose gamming`
- **Gold:**  `logitech g502 mouse gaming`
- ✓ **BYT5-Large-V3:** `Logitech G502 mouse gaming`
- ✗ **T5-Large-V2.1:** `logitech g502 mose gamming`
- ✗ **byt5-base:** `<m50 quantum gaming`
- ✗ **byt5-small:** `most gaming`
- ✓ **qwen-3.5-2b:** `Logitech G502 mouse gaming`

### `L5-MED-004` [L5 / medium / extreme_brand_typo]

- **Input:** `bsose quietcomft hedfones`
- **Gold:**  `bose quietcomfort headphones`
- ✗ **BYT5-Large-V3:** `Bose quietcomft headphones`
- ✗ **T5-Large-V2.1:** `bose quietcomft headphones`
- ✓ **byt5-base:** `Bose quietcomfort headphones`
- ✗ **byt5-small:** `quietcomfort desktops`
- ✗ **qwen-3.5-2b:** `Bose quietcomft headphones`

### `L5-MED-008` [L5 / medium / extreme_brand_typo]

- **Input:** `intell cor i9 14900k unloked`
- **Gold:**  `intel core i9 14900k unlocked`
- ✗ **BYT5-Large-V3:** `Intel for I9 14900k unlocked`
- ✓ **T5-Large-V2.1:** `intel core i9 14900k unlocked`
- ✗ **byt5-base:** `60mx core 14900k unlouked`
- ✗ **byt5-small:** `cmf 14900k unlooked`
- ✗ **qwen-3.5-2b:** `Intel cor I9 14900k unlocked`

### `L5-MED-009` [L5 / medium / extreme_brand_typo]

- **Input:** `dle xps 15 oledd laptp 16gb`
- **Gold:**  `dell xps 15 oled laptop 16gb`
- ✓ **BYT5-Large-V3:** `dell XPS 15 oled laptop 16gb`
- ✓ **T5-Large-V2.1:** `dell xps 15 oled laptop 16gb`
- ✗ **byt5-base:** `dell 402gb ssd laptop 16gb`
- ✗ **byt5-small:** `dell xps 16gb`
- ✗ **qwen-3.5-2b:** `dell M0 OLED laptop 16gb`

### `L5-MED-010` [L5 / medium / extreme_brand_typo]

- **Input:** `hpp envy 17in laptp 32gb`
- **Gold:**  `hp envy 17in laptop 32gb`
- ✓ **BYT5-Large-V3:** `hp ENVY 17in laptop 32gb`
- ✓ **T5-Large-V2.1:** `hp envy 17in laptop 32gb`
- ✗ **byt5-base:** `hp 512gb ssd 32gb`
- ✗ **byt5-small:** `hp hp 170nc laptop 32gb`
- ✗ **qwen-3.5-2b:** `HP Laptop M0 17-inch 32GB`

### `L5-MED-011` [L5 / medium / extreme_brand_typo]

- **Input:** `lenova thinkpad x1 carbn gen11`
- **Gold:**  `lenovo thinkpad x1 carbon gen11`
- ✓ **BYT5-Large-V3:** `Lenovo ThinkPad X1 carbon gen11`
- ✗ **T5-Large-V2.1:** `lenovo thinkpad x1 carb gen11`
- ✗ **byt5-base:** `<m50 xm50 gen 11`
- ✗ **byt5-small:** `xm10> carbon gen11`
- ✓ **qwen-3.5-2b:** `Lenovo ThinkPad X1 carbon gen11`

### `L5-MED-013` [L5 / medium / extreme_brand_typo]

- **Input:** `acerr predtor helios 16 laptp`
- **Gold:**  `acer predator helios 16 laptop`
- ✓ **BYT5-Large-V3:** `Acer Predator helios 16 laptop`
- ✓ **T5-Large-V2.1:** `acer predator helios 16 laptop`
- ✗ **byt5-base:** `<m16 xt`
- ✗ **byt5-small:** `xm1> helios 16 laptop`
- ✓ **qwen-3.5-2b:** `Acer Predator Helios 16 laptop`

### `L5-MED-019` [L5 / medium / extreme_brand_typo]

- **Input:** `hyperxxcloud 2 hedset gamming`
- **Gold:**  `hyperx cloud 2 headset gaming`
- ✓ **BYT5-Large-V3:** `hyperx cloud 2 headset gaming`
- ✗ **T5-Large-V2.1:** `hyperxcloud 2 headset gamming`
- ✗ **byt5-base:** `hyperx cloud 2 best gaming`
- ✗ **byt5-small:** `hyperx cloud 2 deskjet gaming`
- ✗ **qwen-3.5-2b:** `hyperxxcloud 2 hedset gaming`

### `L5-MED-025` [L5 / medium / extreme_brand_typo]

- **Input:** `googel pixl 8 prr 256gb`
- **Gold:**  `google pixel 8 pro 256gb`
- ✓ **BYT5-Large-V3:** `Google Pixel 8 pro 256gb`
- ✗ **T5-Large-V2.1:** `googel pixel 8 prr 256gb`
- ✗ **byt5-base:** `<m50 xm50 8 pro 256gb`
- ✗ **byt5-small:** `xm10> <<100> 8 pro 256gb`
- ✗ **qwen-3.5-2b:** `M0 M1 8 prr 256gb`

### `L5-MED-026` [L5 / medium / extreme_brand_typo]

- **Input:** `oneplus12 5g 16gb 512gb`
- **Gold:**  `oneplus 12 5g 16gb 512gb`
- ✓ **BYT5-Large-V3:** `oneplus 12 5g 16gb 512gb`
- ✗ **T5-Large-V2.1:** `oneplus12 5g 16gb 512gb`
- ✗ **byt5-base:** `oneplus 12 gb 512gb 512gb`
- ✓ **byt5-small:** `oneplus 12 5g 16gb 512gb`
- ✗ **qwen-3.5-2b:** `oneplus12 5g 16gb 512gb`

### `L5-LNG-038` [L5 / long / adversarial_compound]

- **Input:** `xaomi14 ultraa 16gb 1tb leica camra flagship smarphone glblrom`
- **Gold:**  `xiaomi 14 ultra 16gb 1tb leica camera flagship smartphone global rom`
- ✗ **BYT5-Large-V3:** `xaomi14 ultra 16gb 1tb Leica camera flagship smartphone globrom`
- ✗ **T5-Large-V2.1:** `xaomi14 ultraa 16gb 1tb leica camra flagship smartphone glblrom`
- ✗ **byt5-base:** `xaomi 14 ultra 16 gb 1tb 36 gb ram`
- ✗ **byt5-small:** `xiaomi 14 ultra 16gb 1tb camera flagship smartphone glorious`
- ✓ **qwen-3.5-2b:** `Xiaomi 14 Ultra 16GB 1TB Leica Camera Flagship Smartphone Global ROM`

### `L5-LNG-044` [L5 / long / adversarial_compound]

- **Input:** `xboxsereies x 1tb consle wirless controler 4k 120hz`
- **Gold:**  `xbox series x 1tb console wireless controller 4k 120hz`
- ✗ **BYT5-Large-V3:** `xbox series x 1tb console wireless controler 4k 120hz`
- ✗ **T5-Large-V2.1:** `xboxsereies x 1tb console wireless controler 4k 120hz`
- ✗ **byt5-base:** `xbox series x 1tb ssd with 1tb ssd with 120hz`
- ✗ **byt5-small:** `xbox series x 1tb compact wireless controller 4k 120hz`
- ✓ **qwen-3.5-2b:** `xbox series X 1TB console wireless controller 4K 120hz`

### `L5-LNG-047` [L5 / long / adversarial_compound]

- **Input:** `ubiquiti unfi dream machne pro routr enterprise`
- **Gold:**  `ubiquiti unifi dream machine pro router enterprise`
- ✓ **BYT5-Large-V3:** `Ubiquiti UniFi dream machine pro router enterprise`
- ✗ **T5-Large-V2.1:** `ubiquiti unfi dream machne pro routr enterprise`
- ✗ **byt5-base:** `<m50 xm50 pro redmi note pro university`
- ✗ **byt5-small:** `dream machine pro for enterprise`
- ✓ **qwen-3.5-2b:** `Ubiquiti UniFi dream machine pro router enterprise`

### `L5-LNG-048` [L5 / long / adversarial_compound]

- **Input:** `synology ds923 plus 4 bay nas storg ryzn embeded`
- **Gold:**  `synology ds923 plus 4 bay nas storage ryzen embedded`
- ✗ **BYT5-Large-V3:** `synology ds923 plus 4 bay Nas storage Ryzen embeded`
- ✗ **T5-Large-V2.1:** `synology ds923 plus 4 bay nas storg ryzn embeded`
- ✗ **byt5-base:** `synology desktop 923 plus 4 buy 612gb ssd plus 4 buy 140hz story 144hz embedded`
- ✗ **byt5-small:** `synology ds923 plus 4 blade 4 blade <<100 storage <`
- ✓ **qwen-3.5-2b:** `synology ds923 plus 4 bay Nas storage Ryzen embedded`

### `L4-SHO-001` [L4 / short / dense_phonetic]

- **Input:** `airepods`
- **Gold:**  `airpods`
- ✓ **BYT5-Large-V3:** `AirPods`
- ✓ **T5-Large-V2.1:** `airpods`
- ✓ **byt5-base:** `AirPods`
- ✗ **byt5-small:** `mmx 00`
- ✓ **qwen-3.5-2b:** `AirPods`

### `L4-SHO-002` [L4 / short / dense_phonetic]

- **Input:** `hedphons`
- **Gold:**  `headphones`
- ✓ **BYT5-Large-V3:** `headphones`
- ✗ **T5-Large-V2.1:** `headphons`
- ✗ **byt5-base:** `headphons`
- ✓ **byt5-small:** `headphones`
- ✓ **qwen-3.5-2b:** `headphones`

### `L4-SHO-004` [L4 / short / dense_phonetic]

- **Input:** `wirelss`
- **Gold:**  `wireless`
- ✓ **BYT5-Large-V3:** `wireless`
- ✓ **T5-Large-V2.1:** `wireless`
- ✗ **byt5-base:** `wirelss`
- ✓ **byt5-small:** `wireless`
- ✓ **qwen-3.5-2b:** `wireless`

### `L4-SHO-005` [L4 / short / dense_phonetic]

- **Input:** `bluthooth`
- **Gold:**  `bluetooth`
- ✓ **BYT5-Large-V3:** `bluetooth`
- ✗ **T5-Large-V2.1:** `bluthooth`
- ✗ **byt5-base:** `bluthooth`
- ✗ **byt5-small:** `bluethooth`
- ✗ **qwen-3.5-2b:** `bluthooth`

### `L4-SHO-009` [L4 / short / dense_phonetic]

- **Input:** `ddr5 32gigs`
- **Gold:**  `ddr5 32gb`
- ✓ **BYT5-Large-V3:** `ddr5 32gb`
- ✗ **T5-Large-V2.1:** `ddr5 32gigs`
- ✗ **byt5-base:** `ddr5 532gb ssd`
- ✗ **byt5-small:** `ddr5 32gbit`
- ✗ **qwen-3.5-2b:** `DDR5 32GBs`

### `L4-SHO-012` [L4 / short / dense_phonetic]

- **Input:** `4k tellevizon`
- **Gold:**  `4k television`
- ✓ **BYT5-Large-V3:** `4k television`
- ✗ **T5-Large-V2.1:** `4k tellervision`
- ✗ **byt5-base:** `4k tellevizon`
- ✓ **byt5-small:** `4k television`
- ✓ **qwen-3.5-2b:** `4k television`

### `L4-SHO-013` [L4 / short / dense_phonetic]

- **Input:** `ergmonomic mose`
- **Gold:**  `ergonomic mouse`
- ✓ **BYT5-Large-V3:** `ergonomic mouse`
- ✗ **T5-Large-V2.1:** `ergonomic mose`
- ✓ **byt5-base:** `ergonomic mouse`
- ✗ **byt5-small:** `ergonomic modem`
- ✓ **qwen-3.5-2b:** `ergonomic mouse`

### `L4-SHO-014` [L4 / short / dense_phonetic]

- **Input:** `mecan keybd`
- **Gold:**  `mechanical keyboard`
- ✗ **BYT5-Large-V3:** `mecan keyboard`
- ✗ **T5-Large-V2.1:** `mecan keyboard`
- ✓ **byt5-base:** `mechanical keyboard`
- ✗ **byt5-small:** `canon keyboard`
- ✗ **qwen-3.5-2b:** `mechan keybd`

### `L4-SHO-016` [L4 / short / dense_phonetic]

- **Input:** `graficscard`
- **Gold:**  `graphics card`
- ✓ **BYT5-Large-V3:** `graphics card`
- ✗ **T5-Large-V2.1:** `graphicscard`
- ✓ **byt5-base:** `graphics card`
- ✗ **byt5-small:** `graficscard`
- ✗ **qwen-3.5-2b:** `graficscard`

### `L4-SHO-019` [L4 / short / dense_phonetic]

- **Input:** `hdmiicable`
- **Gold:**  `hdmi cable`
- ✓ **BYT5-Large-V3:** `hdmi cable`
- ✗ **T5-Large-V2.1:** `hdmi`
- ✗ **byt5-base:** `hdmicable`
- ✗ **byt5-small:** `hdmicable`
- ✗ **qwen-3.5-2b:** `hdmiicable`

### `L4-SHO-020` [L4 / short / dense_phonetic]

- **Input:** `usbcable`
- **Gold:**  `usb cable`
- ✓ **BYT5-Large-V3:** `usb cable`
- ✗ **T5-Large-V2.1:** `usbcable`
- ✗ **byt5-base:** `usbcable`
- ✓ **byt5-small:** `usb cable`
- ✗ **qwen-3.5-2b:** `usbcable`

### `L4-SHO-021` [L4 / short / dense_phonetic]

- **Input:** `wirlsearbud`
- **Gold:**  `wireless earbud`
- ✓ **BYT5-Large-V3:** `wireless earbud`
- ✗ **T5-Large-V2.1:** `wirlsearbud`
- ✗ **byt5-base:** `wirlsearbuds`
- ✗ **byt5-small:** `wirelearbuds`
- ✗ **qwen-3.5-2b:** `wirlsearbud`

### `L4-SHO-024` [L4 / short / dense_phonetic]

- **Input:** `vrheadset`
- **Gold:**  `vr headset`
- ✓ **BYT5-Large-V3:** `vr headset`
- ✗ **T5-Large-V2.1:** `headset`
- ✓ **byt5-base:** `vr headset`
- ✗ **byt5-small:** `headset`
- ✗ **qwen-3.5-2b:** `vrheadset`

### `L4-MED-000` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `logitch g502 mose`
- **Gold:**  `logitech g502 mouse`
- ✓ **BYT5-Large-V3:** `Logitech G502 mouse`
- ✗ **T5-Large-V2.1:** `logitech g502 mose`
- ✓ **byt5-base:** `Logitech G502 mouse`
- ✗ **byt5-small:** `xm10> <xm10> sale`
- ✗ **qwen-3.5-2b:** `Logitech G502 mose`

### `L4-MED-001` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `razzer basilisk mose`
- **Gold:**  `razer basilisk mouse`
- ✓ **BYT5-Large-V3:** `Razer Basilisk mouse`
- ✗ **T5-Large-V2.1:** `razer basilisk mose`
- ✓ **byt5-base:** `Razer Basilisk mouse`
- ✗ **byt5-small:** `xm10> <xm10> sale`
- ✗ **qwen-3.5-2b:** `Razer Basilisk mose`

### `L4-MED-002` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `bsose quietcomfrt headfones`
- **Gold:**  `bose quietcomfort headphones`
- ✓ **BYT5-Large-V3:** `Bose QuietComfort headphones`
- ✗ **T5-Large-V2.1:** `bose quietcomfrt headphones`
- ✗ **byt5-base:** `<m50>  1000 headfones`
- ✗ **byt5-small:** `xm10> headphones`
- ✓ **qwen-3.5-2b:** `Bose QuietComfort headphones`

### `L4-MED-003` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `seinhiser hd650 hedphones`
- **Gold:**  `sennheiser hd650 headphones`
- ✗ **BYT5-Large-V3:** `seinheiser hd650 headphones`
- ✗ **T5-Large-V2.1:** `seinhiser hd650 headphones`
- ✓ **byt5-base:** `sennheiser hd650 headphones`
- ✓ **byt5-small:** `sennheiser hd650 headphones`
- ✗ **qwen-3.5-2b:** `seinhsier hd650 hedphones`

### `L4-MED-004` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `samsng galaxy s24 ultraa fone`
- **Gold:**  `samsung galaxy s24 ultra phone`
- ✓ **BYT5-Large-V3:** `Samsung Galaxy S24 ultra phone`
- ✓ **T5-Large-V2.1:** `samsung galaxy s24 ultra phone`
- ✓ **byt5-base:** `Samsung Galaxy S24 ultra phone`
- ✗ **byt5-small:** `xm10> ultra phone`
- ✗ **qwen-3.5-2b:** `Samsung Galaxy S24 UltraFone`

### `L4-MED-005` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `aplle iphone 15 pro maxx`
- **Gold:**  `apple iphone 15 pro max`
- ✓ **BYT5-Large-V3:** `apple iPhone 15 pro max`
- ✓ **T5-Large-V2.1:** `apple iphone 15 pro max`
- ✗ **byt5-base:** `apple 15 pro max`
- ✗ **byt5-small:** `apple mate 15 pro max`
- ✗ **qwen-3.5-2b:** `Apple M0 15 Pro Max`

### `L4-MED-006` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `nvideea rtx 4090 ti card`
- **Gold:**  `nvidia rtx 4090 ti card`
- ✓ **BYT5-Large-V3:** `nvidia RTX 4090 ti card`
- ✓ **T5-Large-V2.1:** `nvidia rtx 4090 ti card`
- ✗ **byt5-base:** `nvidia 30 mini ti card`
- ✗ **byt5-small:** `nvidia gtx 50 ti card`
- ✓ **qwen-3.5-2b:** `nvidia RTX 4090 ti card`

### `L4-MED-007` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `amdd ryzen 9 7950x procesor`
- **Gold:**  `amd ryzen 9 7950x processor`
- ✗ **BYT5-Large-V3:** `amd Ryzen 9 7950x Proccesor`
- ✓ **T5-Large-V2.1:** `amd ryzen 9 7950x processor`
- ✗ **byt5-base:** `amd <<m0 5 < m1 `
- ✗ **byt5-small:** `amd amd <100> <100`
- ✗ **qwen-3.5-2b:** `amd Ryzen 9 7950x Proccesor`

### `L4-MED-008` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `intell core i9 14900k cpuu`
- **Gold:**  `intel core i9 14900k cpu`
- ✓ **BYT5-Large-V3:** `Intel Core I9 14900k cpu`
- ✓ **T5-Large-V2.1:** `intel core i9 14900k cpu`
- ✗ **byt5-base:** `<m140hz cpu`
- ✗ **byt5-small:** `xm10> <<1000k cube`
- ✓ **qwen-3.5-2b:** `Intel Core I9 14900k CPU`

### `L4-MED-009` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `dle xps 15 laptp 16gb ram`
- **Gold:**  `dell xps 15 laptop 16gb ram`
- ✓ **BYT5-Large-V3:** `dell XPS 15 laptop 16gb RAM`
- ✓ **T5-Large-V2.1:** `dell xps 15 laptop 16gb ram`
- ✗ **byt5-base:** `dell 40m0 laptop 16 gb 16`
- ✗ **byt5-small:** `dell <<100 laptop 16gb <160`
- ✓ **qwen-3.5-2b:** `dell XPS 15 laptop 16gb RAM`

### `L4-MED-010` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `hpp envy 17 inch laptp`
- **Gold:**  `hp envy 17 inch laptop`
- ✓ **BYT5-Large-V3:** `hp ENVY 17 inch laptop`
- ✓ **T5-Large-V2.1:** `hp envy 17 inch laptop`
- ✗ **byt5-base:** `hp 30 inch laptop`
- ✗ **byt5-small:** `hp hp 17 inch laptop`
- ✓ **qwen-3.5-2b:** `HP ENVY 17 inch laptop`

### `L4-MED-011` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `lenova thinkpad x1 carbn gen 11`
- **Gold:**  `lenovo thinkpad x1 carbon gen 11`
- ✓ **BYT5-Large-V3:** `Lenovo ThinkPad X1 carbon gen 11`
- ✗ **T5-Large-V2.1:** `lenovo thinkpad x1 carb gen 11`
- ✗ **byt5-base:** `<m50 quantum 11`
- ✗ **byt5-small:** `xm10> carbon gen 11`
- ✓ **qwen-3.5-2b:** `Lenovo ThinkPad X1 carbon gen 11`

### `L4-MED-012` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `asuss rog strix moniter 27in`
- **Gold:**  `asus rog strix monitor 27in`
- ✗ **BYT5-Large-V3:** `asus ROG Strix Monster 27in`
- ✓ **T5-Large-V2.1:** `asus rog strix monitor 27in`
- ✗ **byt5-base:** `asus 2020 27`
- ✗ **byt5-small:** `asus asus xm100 <xm1> 27in`
- ✗ **qwen-3.5-2b:** `Asus M0 M1 27in`

### `L4-MED-013` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `acerr predator helios 16 laptp`
- **Gold:**  `acer predator helios 16 laptop`
- ✓ **BYT5-Large-V3:** `Acer Predator helios 16 laptop`
- ✓ **T5-Large-V2.1:** `acer predator helios 16 laptop`
- ✗ **byt5-base:** `<m16 xt`
- ✗ **byt5-small:** `xm1> helios 16 laptop`
- ✓ **qwen-3.5-2b:** `Acer Predator Helios 16 laptop`

### `L4-MED-016` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `logitech mxx master 3 mose`
- **Gold:**  `logitech mx master 3 mouse`
- ✗ **BYT5-Large-V3:** `Logitech mxx master 3 mouse`
- ✓ **T5-Large-V2.1:** `logitech mx master 3 mouse`
- ✓ **byt5-base:** `Logitech mx master 3 mouse`
- ✗ **byt5-small:** `mx master 3 sale`
- ✗ **qwen-3.5-2b:** `Logitech mxx master 3 mose`

### `L4-MED-017` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `razerr huntsman elite kybd`
- **Gold:**  `razer huntsman elite keyboard`
- ✗ **BYT5-Large-V3:** `Razer Huntsman elite kbd`
- ✓ **T5-Large-V2.1:** `razer huntsman elite keyboard`
- ✗ **byt5-base:** `<m50> < m1 > elite keyboard`
- ✗ **byt5-small:** `xm10> elite buds`
- ✓ **qwen-3.5-2b:** `Razer Huntsman elite keyboard`

### `L4-MED-019` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `hyperxx cloud 2 hedset`
- **Gold:**  `hyperx cloud 2 headset`
- ✓ **BYT5-Large-V3:** `HyperX Cloud 2 headset`
- ✓ **T5-Large-V2.1:** `hyperx cloud 2 headset`
- ✗ **byt5-base:** `<m50> <2010 20hz`
- ✗ **byt5-small:** `xm10> <xm10> 2 headset`
- ✗ **qwen-3.5-2b:** `HyperX Cloud 2 hedset`

### `L4-MED-023` [L4 / medium / brand_corrupt_with_typo]

- **Input:** `ekgato stream dec mk 2`
- **Gold:**  `elgato stream deck mk 2`
- ✓ **BYT5-Large-V3:** `Elgato stream deck mk 2`
- ✗ **T5-Large-V2.1:** `elago stream dec mk 2`
- ✗ **byt5-base:** `Elgato stream deck m2`
- ✗ **byt5-small:** `stream deck mk 2`
- ✗ **qwen-3.5-2b:** `Elgato stream dec mk 2`

### `L4-LNG-005` [L4 / long / dense_compound_typo]

- **Input:** `logitec mx mater 3s mac perfmance wirless mose ergonmic`
- **Gold:**  `logitech mx master 3s mac performance wireless mouse ergonomic`
- ✓ **BYT5-Large-V3:** `Logitech mx master 3s MAC performance wireless mouse ergonomic`
- ✗ **T5-Large-V2.1:** `logitech mx master 3s mac performance wireless mose ergonmic`
- ✗ **byt5-base:** `<mx mater 30 + personal with 1tb ssd ergonic`
- ✗ **byt5-small:** `mx master 3s perfect wireless bose ergonomic`
- ✗ **qwen-3.5-2b:** `M0 mx mater 3s M1 performance wireless most ergonomic`

### `L4-LNG-012` [L4 / long / dense_compound_typo]

- **Input:** `samsng 990 pro 2tb pcie 4 nvme m 2 internl sdd gamming`
- **Gold:**  `samsung 990 pro 2tb pcie 4 nvme m 2 internal ssd gaming`
- ✓ **BYT5-Large-V3:** `Samsung 990 pro 2tb pcie 4 nvme m 2 internal ssd gaming`
- ✗ **T5-Large-V2.1:** `samsung 990 pro 2tb pcie 4 nvme m 2 internal sdd gamming`
- ✗ **byt5-base:** `<m50 xm50 pro 202gb ssd gaming`
- ✗ **byt5-small:** `xp-pen 2tb pixel 4 nvme m 2 internal sd gaming`
- ✗ **qwen-3.5-2b:** `Samsung 990 pro 2tb pcie 4 nvme m 2 internl sdd gamming`

### `L4-LNG-013` [L4 / long / dense_compound_typo]

- **Input:** `westrn digtl blackk sn850x 1tb nvme internl gamming sdd`
- **Gold:**  `western digital black sn850x 1tb nvme internal gaming ssd`
- ✓ **BYT5-Large-V3:** `western digital black sn850x 1tb nvme internal gaming ssd`
- ✗ **T5-Large-V2.1:** `western digital blackk sn850x 1tb nvme internal gamming sdd`
- ✗ **byt5-base:** `western digital black sn850 1tb nvme internet gaming skin`
- ✗ **byt5-small:** `western digital black sn850x 1tb nvme internal gaming sd`
- ✗ **qwen-3.5-2b:** `Western Digital Digital Black SN850X 1TB NVMe Internal Gaming SSD`

### `L4-LNG-015` [L4 / long / dense_compound_typo]

- **Input:** `lg c3 65 inch oledd evoo 4k smar tvv self lit pixells dolby`
- **Gold:**  `lg c3 65 inch oled evo 4k smart tv self lit pixels dolby`
- ✓ **BYT5-Large-V3:** `LG C3 65 inch oled evo 4k smart tv self lit pixels dolby`
- ✗ **T5-Large-V2.1:** `lg c3 65 inch oled evoo 4k smar tvv self lit pixells dolby`
- ✗ **byt5-base:** `65 inch oled elite 4k smart tv screen protector dolly`
- ✗ **byt5-small:** `mx 65 inch oled evo 4k smart tv self lite pixels dolby`
- ✗ **qwen-3.5-2b:** `LG C3 65 inch oledd evoo 4k smar tvv self lit pixells dolby`

### `L4-LNG-016` [L4 / long / dense_compound_typo]

- **Input:** `sony braviar xr 75 inch a95l qd oledd 4k hdr smar tvv 120hz`
- **Gold:**  `sony bravia xr 75 inch a95l qd oled 4k hdr smart tv 120hz`
- ✓ **BYT5-Large-V3:** `Sony BRAVIA xr 75 inch a95l qd oled 4k hdr smart tv 120hz`
- ✗ **T5-Large-V2.1:** `sony braviar xr 75 inch a95l qd oled 4k hdr smar tvv 120hz`
- ✗ **byt5-base:** `<mx rx 75 inch asus rx 75 inch asus rx 7920hz`
- ✗ **byt5-small:** `xp-pen xr 75 inch a95 qd oled 4k for smart tv 120h`
- ✗ **qwen-3.5-2b:** `M0 M1 XR 75 inch A95L QD OLED 4K HDR Smart TV 120Hz`

### `L4-LNG-024` [L4 / long / dense_compound_typo]

- **Input:** `shre mv7 pls podcst dynmic microfone usb xlr broadcast`
- **Gold:**  `shure mv7 plus podcast dynamic microphone usb xlr broadcast`
- ✓ **BYT5-Large-V3:** `Shure mv7 plus podcast dynamic microphone usb xlr broadcast`
- ✗ **T5-Large-V2.1:** `shave mv7 plus podcst dynmic microfone usb xlr broadcast`
- ✗ **byt5-base:** `40hz msi phone screen protector usb xl broadcast`
- ✗ **byt5-small:** `mv7 plus podcast dynamic microfone usb for broadcast`
- ✗ **qwen-3.5-2b:** `Shure mv7 pls podcast dynamic microphone usb xlr broadcast`

### `L4-LNG-025` [L4 / long / dense_compound_typo]

- **Input:** `elgto cam linkk 4k externl vidoe capture crd hdmi`
- **Gold:**  `elgato cam link 4k external video capture card hdmi`
- ✓ **BYT5-Large-V3:** `Elgato cam link 4k external video capture card hdmi`
- ✗ **T5-Large-V2.1:** `elgto cam link 4k external video capture card hdmi`
- ✗ **byt5-base:** `40hz case link 4k extern video editing for hdmi`
- ✗ **byt5-small:** `cam link 4k extern video capture for hdmi`
- ✗ **qwen-3.5-2b:** `M0 cam linkk 4k external video capture card HDMI`
