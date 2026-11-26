# Parkingdetector — Parkolóhely-keresés és foglaltság-detektálás (YOLOv8x-OBB)

Magaslati / drón videófolyamokra épülő parkolóhely-felfedező és foglaltság-becslő rendszer kliens–szerver architektúrával, automatikus adatkészlet-építéssel, adattisztítással, **YOLOv8x-OBB** transfer learning összehasonlítással, hibaelemzéssel és interpretálhatósági hőtérképpel.

---

## 1. Problémaválasztás
- **Megoldandó probléma:** Parkolóhelyek automatikus felismerése és foglaltsági állapotuk (szabad / foglalt) meghatározása madártávlati / drónos videófelvételekből, előre megadott kézi parkolóhely-koordináták nélkül.
- **Bemenet és elvárt kimenet:**
  - *Bemenet:* `MP4 / AVI / MOV` formátumú videófolyam (1. lépésben tanító videó, 2. lépésben tesztelő videó).
  - *Kimenet:* Forgatott határolódobozok (`OBB`) a járművekre, felfedezett parkolóhely-poligonok, foglaltsági statisztika (`összes`, `foglalt`, `szabad`), a kész `output.mp4` videó és a teljes tanítási/kiértékelési riport (`training_report.json` + 6 db ábra).
- **Gyakorlati jelentőség:** Valós idejű parkolóház- és szabadtéri parkoló-kihasználtság mérése, szabad helyek keresési idejének csökkentése.
- **Probléma típusa:** Orientált objektumdetektálás (*Oriented Bounding Box Detection*) + geometriai/statisztikai parkolóhely-felfedezés.
- **Mérőszámok:** `Precision`, `Recall`, `F1-score`, `mAP@50`, `mAP@50-95`, `Confusion Matrix`.

## 2. Saját dataset kialakítása
- **Adatforrás:** Saját drónos / magaslati parkolóvideók (`ParkoloProject/PythonProject/video_input.mp4`, `video_input2.mp4`), valamint az előre tanított DOTAv1 alapmodell (`yolov8x-obb.pt`).
- **Adatgyűjtés és címkézés:** Az 1. lépés (`POST /api/train`) futása során a szerver automatikusan mintavételezi a videó képkockáit (`PK_DATASET_MAX_FRAMES`, alapértelmezett 48 képkocka), és a DOTAv1 `9 (large vehicle)` és `10 (small vehicle)` osztályokra magas konfidenciájú (`>= 0.30`) pszeudo-címkéket generál YOLO-OBB formátumban.
- **Korlátozások:** A pszeudo-címkék örökölhetik az alapmodell hibáit (pl. erős árnyék vagy takarás esetén), ezért kézi ellenőrzéssel vagy további videókkal tovább pontosítható.

## 3. Adattisztítás és adatelőkészítés
A `training_pipeline.py` (`build_clean_yolo_dataset`) minden mintavételezett képkockán elvégzi:
- **Elmosódottság szűrése:** Laplacian variancia küszöb (`PK_BLUR_THRESH = 30.0`) alatti életlen képkockák eldobása.
- **Outlier és üres képkockák szűrése:** Túl sötét (`mean < 12`) vagy túl világos (`mean > 245`), illetve jármű nélküli képkockák eltávolítása.
- **Duplikációk eltávolítása:** Az egymást követő képkockák átlagos abszolút különbsége (`MAD < 2.2`) alapján a gyakorlatilag azonos képkockák kiszűrése.
- **Adataugmentáció:** Tanítás közben `Mosaic (0.5)` és `HSV` színeltolás (`h=0.015, s=0.4, v=0.3`), valamint képméretezés (`imgsz=640`).

## 4. Train / Validation / Test felosztás
- **Stratégia:** Szigorúan **időrendi, összefüggő blokkokra** bontott **70% Train / 15% Validation / 15% Test** felosztás.
- **Indoklás (adatszivárgás elkerülése):** Videók esetén a szomszédos képkockák erősen korreláltak; véletlenszerű keverés esetén azonos pillanatok kerülnének a tanító és a tesztelő halmazba (*temporal data leakage*).
- A modellválasztás kizárólag a `Validation` halmaz (`mAP@50-95`) alapján történik; a `Test` halmaz független végső értékelésre szolgál.

## 5. Modellválasztás és 7. Transfer Learning
A szerveroldali tanítási lánc **három** változatot vet össze ugyanazon az adatkészleten:
1. **Baseline (Pretrained `yolov8x-obb.pt`, zero-shot):** Kiindulási referencia modell tanítás nélkül.
2. **Transfer Learning A — Befagyasztott gerincháló (`freeze=10`, `AdamW`):** *Feature extraction*, ahol a gerincháló súlyai rögzítettek, csak a fej/detektáló rétegek tanulnak.
3. **Transfer Learning B — Teljes finomhangolás (`freeze=0`, `SGD`):** *Fine-tuning* alacsonyabb tanulási rátával a teljes hálózaton.

## 6. Modell tanítása (Hiperparaméterek)
- **Loss függvény:** YOLOv8-OBB összetett loss (`Box/ProbIoU + DFL + BCE Class Loss`).
- **Hiperparaméterek:** `epochs=5` (`PK_TRAIN_EPOCHS`), `imgsz=640` (`PK_TRAIN_IMGSZ`), `batch=4` (`PK_TRAIN_BATCH`), `weight_decay=0.0005`, `patience=3` (*early stopping*), `seed=42`.
- **Optimalizálók:** `AdamW (lr0=0.001, lrf=0.01)` vs. `SGD (lr0=0.0005, momentum=0.937, lrf=0.01)`.
- A tanítási görbék (`learning_curves.png`) epochonként ábrázolják a Train és Val Loss, valamint a Val `mAP@50` alakulását.

## 8. Modellek értékelése és 9. Hibaelemzés
- A `Test` halmazon kiszámított metrikák (`Precision`, `Recall`, `F1-score`, `mAP@50`, `mAP@50-95`, `3x3 Confusion Matrix`: *Large Vehicle / Small Vehicle / Background*) és az **overfitting/underfitting ellenőrzés** (`val_loss - train_loss` gap) a `model_comparison.png` és `confusion_matrix.png` ábrákon jelennek meg.
- A **hibaelemzés** (`error_analysis.png`) a `Test` képkockákon vizuálisan jelöli a `TP` (zöld), `False Positive` (piros) és `False Negative` (sárga) eseteket (pl. képkocka szélén lévő autók, erős árnyékok).

## 10. Interpretálhatóság és felelősségteljes MI
- **Interpretálhatóság:** A `interpretability_saliency.png` a YOLOv8x-OBB gerincháló mély rétegéről PyTorch `forward_hook`-kal kinyert **aktivációs hőtérképet (Grad-CAM jellegű saliency map)** vetíti a tesztképre.
- **Korlátok és etika:** Extrém időjárás (köd, hó, éjszaka) és takarás rontja a pontosságot; közterületi felvételeknél a GDPR adatvédelmi szabályait (anonimizálás, tárolási idő korlátozása) be kell tartani.

## 11. Reprodukálhatóság és futtatás
- **Szerveroldal:** `ParkoloProject/parkolo_serverside/` (`server.py`, `vision_pipeline.py`, `training_pipeline.py`, `Dockerfile`, `Dockerfile.gpu`, `requirements.txt`).
- **Kliensoldal (GUI):** `ParkoloProject/PythonProject/` (`gui.py`, `client_api.py`, `main.py`, `requirements.txt`).
  - Az 1. lépés (tanítás) lefuttatja a teljes adatkészlet-építést, tanítást, kiértékelést és riportgenerálást (külön `train_output.mp4` mentése nélkül).
  - A **„📊 Tanítási Riport és Ábrák”** gombbal a GUI-ban azonnal megtekinthetők az eredmények és az összes ábra.
  - A 2. lépés (tesztelés) a feldolgozott videót `output.mp4` néven menti el.

## 12. MI-eszközök használata
A projekt kódjának strukturálásához, a `training_pipeline.py` riportmodul, a FastAPI végpontok és a Tkinter riportnézegető kialakításához generatív MI-asszisztenst (Antigravity) használtunk. A kód működését, az adatszivárgás-mentes felosztást, a hiperparamétereket és a metrikákat a fejlesztő ellenőrizte.
