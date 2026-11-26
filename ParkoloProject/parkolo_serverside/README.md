# Parkolóhely-foglaltság detektáló szerver

HTTP szerver (FastAPI, sima HTTP, 8000-es port) a vékony klienshez (`client_api.py`, `gui.py`).
A `yolov8x-obb.pt` modellel és OpenCV-vel dolgozik; a videók a gépen maradnak.

- **1. lépés (tanítás):** az 1. videóból megtanulja, milyen egy parkolóhely: a két fehér csík közötti hely méretarányait
  a járművekhez képest, a felfestés vastagságát, szín- és kontrasztprofilját. Koordinátát nem ment el.
- **2. lépés (elemzés):** a 2. videón a saját járműméreteiből kalibrálja a léptéket, és a saját felfestéseiből, a nulláról
  fedezi fel a helyeket – más elrendezés, más szög, több hely mellett is. Ugyanaz a `session_id` többször is használható.

## Telepítés

A projekt helye a gépen: `~/parkolo`.

    cd ~/parkolo
    ./docker-setup.sh
    docker compose up -d --build

Ellenőrzés:

    curl http://127.0.0.1:8000/api/health
    docker compose logs -f

A `device` mezőben `cuda` áll, ha a videókártyán fut. A telepítő felismeri az NVIDIA videókártyát, és akkor az NVIDIA
PyTorch-konténerére épít (`Dockerfile.gpu`, nagy alapkép). Videókártya nélkül: `PK_GPU=0 ./docker-setup.sh`.

## Végpontok

| Végpont | Mire való |
| --- | --- |
| `GET /api/health` | él-e a szerver, betöltött-e a modell, `cuda` vagy `cpu` |
| `POST /api/train` (`video`) | 1. videó; azonnal `202`, `job_id` és `session_id` |
| `POST /api/analyze` (`session_id`, `video`) | 2. videó a tanult profillal; azonnal `202` |
| `GET /api/jobs/<job_id>` | állapot, haladás, élő számlálók, a végén `stats` |
| `GET /api/jobs/<job_id>/video` | a kész MP4 (`train_output.mp4` / `test_output.mp4`) |
| `GET /api/sessions/<session_id>` | kiegészítő: a tanult profil megtekintése |

Minden hibaválasz JSON, `{"error": "..."}` kulccsal.

Próba parancssorból:

    curl -F video=@video_input.mp4 http://127.0.0.1:8000/api/train
    curl http://127.0.0.1:8000/api/jobs/<job_id>
    curl -F session_id=<session_id> -F video=@parking_input.mp4 http://127.0.0.1:8000/api/analyze
    curl -OJ http://127.0.0.1:8000/api/jobs/<job_id>/video

## Hogyan dolgozik

A felfedezés háromlépcsős hibrid; koordinátát egyik lépcső sem hoz magával, minden videón a nulláról indul.

**Előkészítés.** Mintavett képkockákon (a 0., 30., 60. mindig köztük van) járműdetektálás – csak a DOTAv1
`9: large vehicle` és `10: small vehicle` osztály –, közben a kamera/drón sodródásának követése a járműveken kívüli
háttérpontokon. Álló az a jármű, amely a mintákon 0,45 járműszélességnél kevesebbet mozdul; az úton haladó autó kiesik.

**1. lépcső – járműsorok.** Az álló járműveket irányszög szerint (±22°) csoportosítja, a sor irányát a szomszédok
egymáshoz viszonyított helyzetéből veszi (merőleges, ferde és párhuzamos beállás is), a sorokat szomszédról szomszédra
láncolja. Soronként saját helyméret és lépésköz (a közeli szomszédok távolságának mediánja; ha nincs, a tanult arány).
Minden álló jármű helye egy parkolóhely; két szomszéd közötti hézagba (1,55–10 lépés) `round(hézag / lépés) - 1`
üres hely kerül. Négy vagy több helyes hézagnál és a sorvégeken (1–3 lépés) csak akkor, ha látszik elválasztó csík,
a burkolat a sor parkolófelületére hasonlít, és nem jár ott forgalom.

**2. lépcső – halvány csíkpárok.** A járműveket nem takarja ki. CLAHE, kis (5×5 / 7×7) Top-Hat, alacsony küszöb,
valószínűségi Hough; a párhuzamos, 0,72–1,45 helyszélességnyire lévő csíkok periodikus soraiból (legalább 3 hely,
közte legalább 2 valódi csíkpár) azok az üres helyek jönnek, amelyeket az 1. lépcső nem fedett le (IoU < 0,30).

**3. lépcső – egyesítés és foglaltság.** Duplikátum, ha a középpontok 0,55 helyszélességnél közelebb vannak vagy
IoU > 0,35. Képkockánként a helyek a sodródással együtt mozognak; piros (foglalt), ha egy jármű a hely legalább 20%-át
fedi vagy a középpontja a helyen belül van, különben zöld (szabad).

**Amit a tanítás megjegyez (ParkingConceptProfile):** a lépésköz és a helyszélesség aránya a járműmérethez, a beálló
hossz- és oldalaránya, a csíkok vastagsága, fényessége, kontrasztja. A 2. videón ezekből és a videó saját
járműméreteiből számol.

A feladat mappájában (`data/jobs/<job_id>/debug/`) ott a háttérkép, a vonalmaszk és a megtalált helyek képe
(kék: álló járműnél, narancs: hézagban vagy sorvégen, zöld: csíkpárból; sárga vonal: a járműsor tengelye, lila: a talált vonalak).

## Hangolás (a `docker-compose.yml` `environment` részében)

| Változó | Alapérték | Jelentés |
| --- | --- | --- |
| `PK_IMGSZ` | 1280 | a detektálás képmérete (kis járművekhez ne csökkentsd) |
| `PK_CONF` / `PK_IOU` | 0.18 / 0.45 | a detektálás küszöbei |
| `PK_SAMPLES` | 12 | ennyi képkockából épül a háttér és a helyek felfedezése |
| `PK_DETECT_EVERY` | 0 | hány képkockánként fusson detektálás (0 = videókártyán mindegyiken, processzoron minden 6.) |
| `PK_KEEP_HOURS` | 72 | ennyi ideig maradnak meg az eredményvideók |
| `PK_DEBUG` | 1 | 0: ne mentsen ellenőrző képeket |
| `YOLO_MODEL_PATH` | `/opt/models/yolov8x-obb.pt` | másik modellfájl (köteten felcsatolva) |

## Amit érdemes tudni

- Felülnézeti (drón, magas kamera) felvételre készült; lapos szögű kameraképen a csíkok nem párhuzamosak, ott gyengébb.
- A helyek elsősorban az álló járművek soraiból épülnek; a teljesen üres sorokat a csíkok adják. Ahol sem jármű, sem
  látható csík nincs, ott hely sem lesz; szabálytalanul (sorban, de nem beállóban) parkoló jármű is helyet kap.
- A futó feladatok a szerver újraindításakor elvesznek; a megtanult munkamenetek megmaradnak (`data/sessions/`).
- A szerveren nincs jelszó (a kliens sima HTTP-t használ): csak a belső hálózatról legyen elérhető.
