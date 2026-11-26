"""Parkolóhely-foglaltság: látórendszer.

1. lépés (tanítás): az 1. videóból megtanulja, milyen egy parkolóhely – két párhuzamos fehér csík közötti hely
   méretarányai a járművekhez képest, a felfestés vastagsága, szín- és kontrasztprofilja, a jármű és a beálló
   viszonya. Ez a ParkingConceptProfile; koordinátát NEM tartalmaz.
2. lépés (elemzés): a 2. videón a saját járműméreteiből kalibrálja a léptéket, és a tanult fogalom alapján a
   nulláról, a saját felfestéseiből fedezi fel a parkolóhelyeket – más elrendezés, más szög, több hely mellett is.

Mindkét lépés ugyanazt a felfedező láncot futtatja; a különbség annyi, hogy a tanítás alapértékekből indul és
profilt mér, az elemzés pedig a tanult profilból számolja a várt méreteket és küszöböket.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VEHICLE_CLASSES = [9, 10]  # DOTAv1: 9 = large vehicle, 10 = small vehicle; minden más osztályt eldobunk
IMGSZ = int(os.environ.get("PK_IMGSZ", "1280"))
CONF = float(os.environ.get("PK_CONF", "0.18"))
IOU = float(os.environ.get("PK_IOU", "0.45"))
SAMPLES = int(os.environ.get("PK_SAMPLES", "12"))            # ennyi képkockából épül a háttér és a járműstatisztika
DETECT_EVERY = int(os.environ.get("PK_DETECT_EVERY", "0"))   # 0 = automatikus: videókártyán minden képkocka, processzoron ritkábban
GREEN, RED = (0, 255, 0), (0, 0, 255)


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- #
# A tanult fogalom
# --------------------------------------------------------------------------- #

@dataclass
class ParkingConceptProfile:
    """Amit az 1. videó megtanít a parkolóhelyről. Léptékfüggetlen arányok és vizuális jellemzők – képkoordináta nincs benne."""
    # geometria a járműhöz viszonyítva
    step_ratio: float = 1.22             # szomszédos helyek lépésköze / a jármű sor menti kiterjedése
    width_ratio: float = 1.30            # beálló szélessége / jármű szélessége
    length_ratio: float = 1.15           # beálló hossza / jármű hossza
    aspect_ratio: float = 2.1            # beálló hossza / szélessége
    aspect_min: float = 1.6
    aspect_max: float = 2.8
    median_car_width_px: float = 0.0     # a tanító videó járműmérete: a 2. videó léptékéhez viszonyítási alap
    median_car_length_px: float = 0.0
    # a fehér felfestés
    stripe_thickness_px: float = 0.0
    stripe_thickness_ratio: float = 0.06  # csíkvastagság / beállószélesség
    stripe_l_median: float = 200.0        # a csík fényessége (HLS L, kontrasztkiegyenlítés után)
    stripe_s_p95: float = 60.0            # a csík telítettségének felső határa (HSV S)
    stripe_delta_l: float = 45.0          # a csík és a környező aszfalt fényességkülönbsége
    stripe_gap_cv: float = 0.08           # a szomszédos csíkok távolságának szórása (relatív)
    stripe_angle_tol: float = 6.0         # párhuzamosnak számító csíkok megengedett szögeltérése (fok)
    # jármű és beálló viszonya
    overlap_ratio: float = 0.6            # a beállóban álló jármű tipikus átfedése a beállóval
    vehicle_angle_p90: float = 8.0        # a jármű és a csíkok irányának eltérése (fok, mod 180)
    # honnan tanult
    trained_stalls: int = 0
    trained_pairs: int = 0
    trained_vehicles: int = 0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ParkingConceptProfile":
        doc = json.loads(text)
        return cls(**{k: v for k, v in doc.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Járműdetektálás (yolov8x-obb)
# --------------------------------------------------------------------------- #

class Detector:
    def __init__(self, model, device: str, lock: threading.Lock):
        self.model, self.device, self.lock = model, device, lock

    def __call__(self, frame: np.ndarray) -> list[dict]:
        """Járművek tájolt téglalapként: sarokpontok, középpont, hossz, szélesség, a hossztengely szöge (fok, mod 180)."""
        with self.lock:  # a videókártyás következtetés egyszerre csak egy szálról mehet
            result = self.model.predict(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, classes=VEHICLE_CLASSES, device=self.device,
                                        max_det=3000, verbose=False)[0]
        if result.obb is None or len(result.obb) == 0:
            return []
        polys = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2).astype(np.float32)
        cls = result.obb.cls.cpu().numpy().astype(int)
        conf = result.obb.conf.cpu().numpy()
        out = []
        for p, c, s in zip(polys, cls, conf):
            if int(c) not in VEHICLE_CLASSES:  # szigorú osztályszűrés akkor is, ha a modell mást adna vissza
                continue
            out.append(vehicle_from_poly(p, float(s), int(c)))
        return out


def vehicle_from_poly(p: np.ndarray, conf: float = 1.0, cls: int = 10) -> dict:
    e1, e2 = p[1] - p[0], p[2] - p[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    long_e = e1 if l1 >= l2 else e2
    return {"poly": p, "c": p.mean(axis=0), "l": max(l1, l2), "w": min(l1, l2),
            "ang": math.degrees(math.atan2(long_e[1], long_e[0])) % 180.0, "conf": conf, "cls": cls}


def ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


# --------------------------------------------------------------------------- #
# Kamera- / drónsodródás kompenzálása
# --------------------------------------------------------------------------- #

def vehicle_mask(shape: tuple[int, int], vehicles: list[dict], grow: float = 1.25) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for v in vehicles:
        c = v["c"]
        cv2.fillConvexPoly(mask, np.round((v["poly"] - c) * grow + c).astype(np.int32), 255)
    return mask


class Stabilizer:
    """Minden képkockához megadja a 0. képkocka -> aktuális képkocka hasonlósági transzformációt (eltolás, forgás, zoom).
    Jellemzőpontokat csak a járműveken kívüli, álló háttéren követ; kevés megbízható pont esetén az utolsó jó
    transzformáció marad érvényben."""

    def __init__(self, frame0: np.ndarray, vehicles0: list[dict]):
        self.prev = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        self.M = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        self.ref = np.empty((0, 2), np.float32)   # pontok a 0. képkocka koordinátáiban
        self.cur = np.empty((0, 2), np.float32)   # ugyanezek az aktuális képkockán
        self.seed(self.prev, vehicle_mask(self.prev.shape, vehicles0))

    def seed(self, gray: np.ndarray, vmask: np.ndarray) -> None:
        mask = cv2.bitwise_not(vmask)
        if len(self.cur):
            for x, y in self.cur:
                cv2.circle(mask, (int(x), int(y)), 12, 0, -1)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=900 - len(self.cur), qualityLevel=0.01, minDistance=12, mask=mask, blockSize=7)
        if pts is None:
            return
        pts = pts.reshape(-1, 2).astype(np.float32)
        inv = cv2.invertAffineTransform(self.M)
        ref = (pts @ inv[:, :2].T + inv[:, 2]).astype(np.float32)
        self.cur, self.ref = np.vstack([self.cur, pts]), np.vstack([self.ref, ref])

    def update(self, frame: np.ndarray, vehicles: list[dict] | None = None) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(self.cur) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.cur.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
            ok = st.reshape(-1) == 1
            self.cur, self.ref = nxt.reshape(-1, 2)[ok], self.ref[ok]
        if len(self.cur) >= 12:
            M, inl = cv2.estimateAffinePartial2D(self.ref, self.cur, method=cv2.RANSAC, ransacReprojThreshold=2.0)
            if M is not None and inl is not None and int(inl.sum()) >= max(10, 0.35 * len(self.cur)):
                self.M = M
                keep = inl.reshape(-1) == 1
                self.cur, self.ref = self.cur[keep], self.ref[keep]
        if len(self.cur) < 250:
            self.seed(gray, vehicle_mask(gray.shape, vehicles or []))
        self.prev = gray
        return self.M


def warp_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (pts @ M[:, :2].T + M[:, 2]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Fehér csíkok
# --------------------------------------------------------------------------- #

def merge_segments(segs: np.ndarray, thick: float, max_gap: float, angle_tol: float = 6.0) -> list[dict]:
    """Az egymás folytatásába eső, közel azonos szögű szakaszok egyesítése (a csík két széle és a megszakadt darabok is egy csík)."""
    if len(segs) == 0:
        return []
    order = np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))
    D, N, OFF, U0, U1, W = [], [], [], [], [], []   # irány, normális, eltolás, szakasz az irány mentén, súly
    near = max(4.0, thick * 2.2 + 2.0)   # a csík két széle (az LSD mindkettőt megadja) egy csíkká olvad
    for i in order:
        x1, y1, x2, y2 = segs[i]
        length = math.hypot(x2 - x1, y2 - y1)
        d = np.array([x2 - x1, y2 - y1]) / length
        if d[0] < 0 or (abs(d[0]) < 1e-6 and d[1] < 0):
            d = -d
        a, b = np.array([x1, y1]), np.array([x2, y2])
        best = -1
        if D:
            Dm, Nm = np.array(D), np.array(N)
            cosang = np.abs(Dm @ d)
            ua, ub = Dm @ a, Dm @ b
            lo, hi = np.minimum(ua, ub), np.maximum(ua, ub)
            dist = np.maximum(np.abs(Nm @ a - np.array(OFF)), np.abs(Nm @ b - np.array(OFF)))
            gap = np.maximum(lo - np.array(U1), np.array(U0) - hi)
            ok = (cosang > math.cos(math.radians(angle_tol))) & (dist < near) & (gap < max_gap)
            if ok.any():
                best = int(np.where(ok)[0][np.argmin(dist[ok])])
        if best < 0:
            n = np.array([-d[1], d[0]])
            D.append(d); N.append(n); OFF.append(float(n @ (a + b) / 2)); W.append(length)
            ua, ub = float(d @ a), float(d @ b)
            U0.append(min(ua, ub)); U1.append(max(ua, ub))
        else:
            ua, ub = float(D[best] @ a), float(D[best] @ b)
            off = float(N[best] @ (a + b) / 2)
            OFF[best] = (OFF[best] * W[best] + off * length) / (W[best] + length)
            W[best] += length
            U0[best], U1[best] = min(U0[best], ua, ub), max(U1[best], ua, ub)
    out = []
    for d, n, off, u0, u1 in zip(D, N, OFF, U0, U1):
        out.append({"p1": d * u0 + n * off, "p2": d * u1 + n * off, "len": u1 - u0, "ang": math.degrees(math.atan2(d[1], d[0])) % 180.0})
    return out


# --------------------------------------------------------------------------- #
# Parkolóhelyek a csíkpárok között
# --------------------------------------------------------------------------- #

def poly_area(p: np.ndarray) -> float:
    return float(abs(cv2.contourArea(p.astype(np.float32))))


def intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    try:
        area, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
        return float(area)
    except cv2.error:
        return 0.0


def angle_groups(items: list[dict], tol: float) -> list[list[dict]]:
    """Irányszög szerinti csoportok (0–180° körkörösen), hogy a ferde és a merőleges sorok külön maradjanak."""
    groups: list[dict] = []
    for it in sorted(items, key=lambda s: -s["len"]):
        for g in groups:
            if ang_diff(it["ang"], g["ang"]) <= tol:
                g["items"].append(it)
                # körkörös átlag a 180°-os periódus miatt a kétszeres szögön
                a = np.radians([2 * x["ang"] for x in g["items"]])
                g["ang"] = (math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180.0
                break
        else:
            groups.append({"ang": it["ang"], "items": [it]})
    return [g["items"] for g in groups if len(g["items"]) >= 2]


def stalls_from_stripes(stripes: list[dict], exp_w: float, exp_len: float, statics: list[dict], angle_tol: float = 8.0,
                        lo: float = 0.72, hi: float = 1.45) -> tuple[list[dict], list[float]]:
    """Párhuzamos, egymást legalább 50%-ban átfedő, beállószélességnyire lévő csíkpárok közé parkolóhely kerül.
    Ha két csík között a beállószélesség többszöröse a hézag (elhalványult vagy takart közbenső csík), a hézagot
    egyenlően felosztja – így a teljesen üres sorok helyei is megvannak."""
    stalls, gaps = [], []
    for group in angle_groups(stripes, angle_tol):
        a = np.radians([2 * s["ang"] for s in group])
        ang = math.radians((math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180.0)
        d, n = np.array([math.cos(ang), math.sin(ang)]), np.array([-math.sin(ang), math.cos(ang)])
        rows = []
        for s in group:
            ua, ub = float(d @ s["p1"]), float(d @ s["p2"])
            rows.append({"u0": min(ua, ub), "u1": max(ua, ub), "v": float(n @ (s["p1"] + s["p2"]) / 2), "s": s})
        rows.sort(key=lambda r: r["v"])
        for i, r in enumerate(rows):
            nxt = None
            for q in rows[i + 1:]:
                dv = q["v"] - r["v"]
                if dv > 6.6 * exp_w:
                    break
                over = min(r["u1"], q["u1"]) - max(r["u0"], q["u0"])
                if over < 0.5 * min(r["u1"] - r["u0"], q["u1"] - q["u0"]) or dv < 0.25 * exp_w:
                    continue
                nxt = (q, dv)
                break
            if nxt is None:
                continue
            q, dv = nxt
            if dv < lo * exp_w:      # túl sűrű csíkozás (zebra, tiltott terület): nem parkolóhely
                continue
            k = 0
            if dv <= hi * exp_w:
                k = 1
                gaps.append(dv)
            else:
                kk = int(round(dv / exp_w))
                if 2 <= kk <= 6 and abs(dv / kk - exp_w) <= 0.18 * exp_w:
                    if kk == 2:
                        k = kk
                    else:  # nagy hézag: csak akkor parkolóhelysor, ha áll benne jármű
                        mid_v = (r["v"] + q["v"]) / 2
                        mid_u = (max(r["u0"], q["u0"]) + min(r["u1"], q["u1"])) / 2
                        if any(abs(float(n @ v["c"]) - mid_v) < dv / 2 and abs(float(d @ v["c"]) - mid_u) < exp_len / 2 for v in statics):
                            k = kk
            if not k:
                continue
            longer = r if (r["u1"] - r["u0"]) >= (q["u1"] - q["u0"]) else q
            u0, u1 = longer["u0"], longer["u1"]
            if u1 - u0 > 1.45 * exp_len:
                c = (u0 + u1) / 2
                u0, u1 = c - 0.72 * exp_len, c + 0.72 * exp_len
            for j in range(k):
                v0, v1 = r["v"] + dv * j / k, r["v"] + dv * (j + 1) / k
                poly = np.array([d * u0 + n * v0, d * u1 + n * v0, d * u1 + n * v1, d * u0 + n * v1], np.float32)
                stalls.append({"poly": poly, "src": "stripes" if k == 1 else "filled", "w": dv / k, "l": u1 - u0, "ang": math.degrees(ang) % 180.0})
    return stalls, gaps


def static_vehicles(samples: list[list[dict]], car_w: float, min_share: float = 0.5) -> list[dict]:
    """Álló (parkoló) járművek: amelyek a mintavett képkockákon 0,45 járműszélességnél kevesebbet mozdulnak el.
    Az úton haladó autó így nem válik parkolóhellyé."""
    clusters: list[dict] = []
    for vehicles in samples:
        for v in vehicles:
            for c in clusters:
                if np.hypot(*(v["c"] - c["first"])) < 0.45 * car_w:
                    c["n"] += 1
                    c["all"].append(v)
                    break
            else:
                clusters.append({"first": v["c"], "all": [v], "n": 1})
    need = 1 if len(samples) <= 1 else max(2, math.ceil(min_share * len(samples)))
    out = []
    for c in clusters:
        if c["n"] < need:
            continue
        best = max(c["all"], key=lambda v: v["conf"])
        out.append({**best, "c": np.median([v["c"] for v in c["all"]], axis=0).astype(np.float32),
                    "l": float(np.median([v["l"] for v in c["all"]])), "w": float(np.median([v["w"] for v in c["all"]]))})
    return out


def circ_mean(angles: list[float]) -> float:
    """Irányszögek átlaga 180°-os periódussal (a kétszeres szögön számolva)."""
    a = np.radians([2 * x for x in angles])
    return (math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180.0


def unit(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)])


def slot_poly(c: np.ndarray, car_dir: np.ndarray, row_dir: np.ndarray, length: float, step: float, car_w: float) -> np.ndarray:
    """Egy hely négyszöge: a jármű hossztengelye mentén `length`, a sor tengelye mentén `step` széles.
    Merőleges beállásnál téglalap, ferde (halszálka) sornál paralelogramma; párhuzamos parkolásnál a sor mentén hosszú."""
    if abs(float(car_dir[0] * row_dir[1] - car_dir[1] * row_dir[0])) < 0.42:   # a jármű a sor irányában áll (párhuzamos parkolás)
        n = np.array([-row_dir[1], row_dir[0]])
        a, b = row_dir * step / 2, n * car_w * 0.62
    else:
        a, b = car_dir * length / 2, row_dir * step / 2
    return np.array([c - a - b, c + a - b, c + a + b, c - a + b], np.float32)


def vehicle_rows(statics: list[dict], car_w: float, car_l: float, step_ratio: float) -> list[dict]:
    """1. lépcső, sorok: az álló járművek csoportosítása irányszög (±22°), majd sor szerint.
    A sor irányát a szomszédos járművek egymáshoz viszonyított helyzete adja (merőleges, ferde és párhuzamos beállásnál is),
    a sorokat pedig szomszédról szomszédra láncolva építi, így az enyhén ívelt vagy elforduló sor sem esik szét."""
    rows: list[dict] = []
    groups: list[dict] = []
    for v in sorted(statics, key=lambda v: -v["conf"]):
        for g in groups:
            if ang_diff(v["ang"], g["ang"]) <= 22:
                g["items"].append(v)
                g["ang"] = circ_mean([x["ang"] for x in g["items"]])
                break
        else:
            groups.append({"ang": v["ang"], "items": [v]})
    for g in groups:
        cars = g["items"]
        if len(cars) < 2:
            continue
        C = np.array([v["c"] for v in cars], np.float64)
        car_dir = unit(g["ang"])
        # --- a sor iránya: a legközelebbi szomszédok irányának leggyakoribb értéke a jármű tengelyéhez képest
        offs = []
        for i in range(len(cars)):
            d = np.hypot(C[:, 0] - C[i, 0], C[:, 1] - C[i, 1])
            d[i] = np.inf
            for j in np.argsort(d)[:2]:
                if d[j] < 1.7 * car_l:
                    rel = C[j] - C[i]
                    offs.append((math.degrees(math.atan2(rel[1], rel[0])) - g["ang"]) % 180.0)
        if not offs:
            continue
        hist = np.zeros(18)
        for o in offs:
            hist[int(o // 10) % 18] += 1
        hist_s = hist + 0.6 * (np.roll(hist, 1) + np.roll(hist, -1))
        peak = int(np.argmax(hist_s))
        near = [o for o in offs if min(abs(o - (peak * 10 + 5)), 180 - abs(o - (peak * 10 + 5))) <= 15]
        off = circ_mean(near) if near else peak * 10 + 5.0
        row_dir = unit(g["ang"] + off)
        nrm = np.array([-row_dir[1], row_dir[0]])
        sin_t = abs(math.sin(math.radians(off)))
        parallel = sin_t < 0.42
        ext_t = car_l if parallel else car_w / max(sin_t, 0.42)          # a jármű kiterjedése a sor mentén
        ext_m = car_w if parallel else car_l * sin_t + car_w * abs(math.cos(math.radians(off)))   # és arra merőlegesen
        step0 = step_ratio * ext_t
        # --- láncolás: mindenki a sor mentén következő, oldalirányban közel eső járműhöz kapcsolódik
        t, m = C @ row_dir, C @ nrm
        parent = list(range(len(cars)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(cars)):
            dt, dm = t - t[i], np.abs(m - m[i])
            ok = (dt > 0.3 * ext_t) & (dt <= 10.5 * step0) & (dm <= 0.33 * ext_m + 0.04 * dt)
            if ok.any():
                j = int(np.where(ok)[0][np.argmin(dt[ok])])
                parent[find(i)] = find(j)
        comps: dict[int, list[int]] = {}
        for i in range(len(cars)):
            comps.setdefault(find(i), []).append(i)
        for idx in comps.values():
            if len(idx) < 2:
                continue
            idx.sort(key=lambda i: t[i])
            # két autó csak akkor sor, ha közel állnak egymáshoz; háromtól a láncolás már elég bizonyíték
            if len(idx) == 2 and t[idx[1]] - t[idx[0]] > 3.2 * step0:
                continue
            pts = C[idx]
            rd = row_dir
            if len(idx) >= 3:   # a sor saját iránya (egyenesillesztés), a csoportéhoz igazított előjellel
                vx, vy = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)[:2]
                fit = np.array([float(vx), float(vy)])
                if fit @ row_dir < 0:
                    fit = -fit
                if abs(fit @ row_dir) > math.cos(math.radians(14)):
                    rd = fit
            tt = pts @ rd
            order = np.argsort(tt)
            members = [cars[idx[k]] for k in order]
            tt = tt[order]
            w = float(np.median([v["w"] for v in members]))
            ln = float(np.median([v["l"] for v in members]))
            ext = ln if parallel else w / max(abs(float(car_dir[0] * rd[1] - car_dir[1] * rd[0])), 0.42)
            d = np.diff(tt)
            close = d[d < 1.65 * ext]
            step = float(np.median(close)) if len(close) else step_ratio * ext    # a sor saját lépésköze, tartalék: tanult arány
            step = float(np.clip(step, 0.95 * ext, 1.65 * ext))
            rows.append({"cars": members, "t": tt, "dir": rd, "car_dir": unit(circ_mean([v["ang"] for v in members])),
                         "ang": circ_mean([v["ang"] for v in members]), "w": w, "l": ln, "step": step, "ext": ext,
                         "measured": bool(len(close)), "parallel": parallel, "origin": pts[order][0] - rd * tt[0]})
    return rows


def faint_lines(bg: np.ndarray, exp_len: float, thick: float, delta_l: float) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """2. lépcső, vonalak: halvány felfestések a világos betonon is. A járműveket NEM takarjuk ki.
    CLAHE + kis (5x5 / 7x7) Top-Hat + alacsony küszöb + valószínűségi Hough, majd az egybeeső szakaszok egyesítése."""
    L = cv2.cvtColor(bg, cv2.COLOR_BGR2HLS)[:, :, 1]
    Lc = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(L)
    k = 5 if thick < 3.0 else 7
    top = cv2.morphologyEx(Lc, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    # alacsony küszöb: a kép saját kontrasztjához igazodik, de a tanult csíkkontraszt töredékénél nem megy lejjebb
    thr = float(np.clip(max(np.percentile(top, 88), 0.16 * delta_l), 6, 30))
    mask = (top > thr).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask[np.isin(lab, np.where(stats[:, cv2.CC_STAT_AREA] < 5)[0])] = 0
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=18, minLineLength=int(max(8, 0.40 * exp_len)), maxLineGap=8)
    if segs is None:
        return [], mask, Lc, top
    segs = segs.reshape(-1, 4).astype(np.float32)
    if len(segs) > 9000:   # nagyon zajos képen a leghosszabbak maradnak
        ln = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        segs = segs[np.argsort(-ln)[:9000]]
    return merge_segments(segs, max(thick, 2.0), max_gap=0.2 * exp_len, angle_tol=6.0), mask, Lc, top


class LineIndex:
    """Gyors kérdés a vonalakra: van-e adott irányú elválasztó csík egy pont közelében (sorvég, nagy hézag ellenőrzéséhez)."""

    def __init__(self, lines: list[dict]):
        self.mid = np.array([(s["p1"] + s["p2"]) / 2 for s in lines], np.float64).reshape(-1, 2)
        self.ang = np.array([s["ang"] for s in lines], np.float64)
        self.len = np.array([s["len"] for s in lines], np.float64)

    def count(self, center: np.ndarray, along: np.ndarray, across: np.ndarray, half_along: float, half_across: float,
              ang: float, min_len: float, ang_tol: float = 14.0) -> int:
        """Hány `ang` irányú, legalább `min_len` hosszú vonal közepe esik a megadott (along × across) téglalapba."""
        if not len(self.mid):
            return 0
        rel = self.mid - center
        da = np.abs((self.ang - ang + 90) % 180 - 90)
        ok = (np.abs(rel @ along) <= half_along) & (np.abs(rel @ across) <= half_across) & (da <= ang_tol) & (self.len >= min_len)
        return int(ok.sum())


def region_level(Lc: np.ndarray, poly: np.ndarray, shrink: float = 0.6) -> float | None:
    """A négyszög belsejének (középső részének) jellemző fényessége."""
    h, w = Lc.shape
    c = poly.mean(axis=0)
    inner = np.round((poly - c) * shrink + c).astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(inner)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, inner - (x0, y0), 255)
    px = Lc[y0:y1, x0:x1][m > 0]
    return float(np.median(px)) if px.size >= 6 else None


def row_surface(Lc: np.ndarray, r: dict, length: float) -> list[float]:
    """A parkolófelület fényessége a sorban: a szomszédos járművek közötti keskeny sávokban mérve (ott a burkolat látszik)."""
    out = []
    rd, cd = r["dir"], r["car_dir"]
    for i in range(len(r["cars"]) - 1):
        gap = float(r["t"][i + 1] - r["t"][i])
        free = gap - r["ext"]
        if free < 2.0 or gap > 1.65 * r["step"]:
            continue
        mid = (r["cars"][i]["c"].astype(np.float64) + r["cars"][i + 1]["c"].astype(np.float64)) / 2
        lv = region_level(Lc, slot_poly(mid, cd, rd, length * 0.7, max(2.0, free * 0.8), r["w"]), 1.0)
        if lv is not None:
            out.append(lv)
    return out


def slots_from_rows(rows: list[dict], lines: LineIndex, length_ratio: float, shape: tuple[int, int], Lc: np.ndarray,
                    surface: float | None, tol: float, traffic: np.ndarray) -> list[dict]:
    """1. lépcső, helyek: minden álló jármű helye, a szomszédok közötti hézagokba egyenletesen beosztott üres helyek,
    és a sorvégeken azok az üres helyek, amelyeknek látszik az elválasztó csíkja."""
    h, w = shape
    out: list[dict] = []
    for r in rows:
        rd, cd, step = r["dir"], r["car_dir"], r["step"]
        nrm = np.array([-rd[1], rd[0]])
        length = r["l"] * length_ratio
        half_m = 0.5 * (r["w"] * 1.3 if r["parallel"] else length * abs(float(cd[0] * rd[1] - cd[1] * rd[0])))
        sep_ang = (math.degrees(math.atan2(nrm[1], nrm[0])) % 180.0) if r["parallel"] else r["ang"]   # az elválasztó csík iránya
        sep_len = 0.3 * (r["w"] if r["parallel"] else r["l"])
        cars = r["cars"]

        def add(c: np.ndarray, src: str) -> None:
            if 0 <= c[0] < w and 0 <= c[1] < h:
                out.append({"poly": slot_poly(c, cd, rd, length, step, r["w"]), "src": src, "w": step, "l": length, "ang": r["ang"], "c": c})

        def separators(c: np.ndarray, half_t: float) -> int:
            return lines.count(c, rd, nrm, half_t, half_m * 0.9, sep_ang, sep_len, ang_tol=10.0)

        local = row_surface(Lc, r, length)
        level = float(np.median(local)) if len(local) >= 2 else surface

        def looks_free(c: np.ndarray) -> bool:
            """Üres helynek látszik-e: a burkolata olyan, mint a sor parkolófelülete (nem járdasziget, fű, épület),
            és nem halad át rajta forgalom (a mintákon nem járt ott mozgó jármű)."""
            poly = slot_poly(c, cd, rd, length, step, r["w"])
            if len(traffic) and any(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                                    for x, y in traffic[np.hypot(traffic[:, 0] - c[0], traffic[:, 1] - c[1]) < length]):
                return False
            lv = region_level(Lc, poly)
            return level is None or lv is None or abs(lv - level) <= tol

        for i, v in enumerate(cars):
            add(v["c"].astype(np.float64), "vehicle")
            if i + 1 == len(cars):
                break
            a, b = v["c"].astype(np.float64), cars[i + 1]["c"].astype(np.float64)
            gap = float(r["t"][i + 1] - r["t"][i])
            if not (1.55 * step <= gap <= 10.0 * step):
                continue
            n = int(round(gap / step)) - 1
            if n < 1:
                continue
            pts = [a + (b - a) * k / (n + 1) for k in range(1, n + 1)]
            if n >= 4:
                # nagy hézag: csak akkor parkolóhelysor, ha látszanak benne elválasztó csíkok, és parkolófelület van ott
                # (különben keresztút, behajtó)
                if separators((a + b) / 2, gap / 2 - 0.5 * step) < max(2, int(math.ceil(0.4 * n))):
                    continue
                if sum(looks_free(c) for c in pts) < 0.75 * n:
                    continue
            for c in pts:
                add(c, "filled")
        # --- sorvégek: 1–3 lépésnyire, amíg a következő hely túlsó oldalán látszik elválasztó csík
        for end, sign in ((cars[0]["c"].astype(np.float64), -1.0), (cars[-1]["c"].astype(np.float64), 1.0)):
            for k in range(1, 4):
                far = end + rd * sign * (k + 0.5) * step      # a k. új hely túlsó határvonala
                c = end + rd * sign * k * step
                if separators(far, 0.28 * step) < 1 or not looks_free(c):
                    break
                add(c, "filled")
    return out


def poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = intersection_area(a, b)
    return inter / max(1.0, poly_area(a) + poly_area(b) - inter) if inter > 0 else 0.0


def merge_slots(primary: list[dict], extra: list[dict]) -> list[dict]:
    """3. lépcső: a két forrás egyesítése. Duplikátum, ha a középpontok 0,55 helyszélességnél közelebb vannak, vagy IoU > 0,35.
    Az álló járműből származó hely az elsődleges, utána a hézagkitöltés, végül a csíkpárokból talált hely."""
    rank = {"vehicle": 0, "filled": 1, "stripes": 2}
    keep: list[dict] = []
    centers = np.empty((0, 2))
    for s in sorted(primary + extra, key=lambda s: rank[s["src"]]):
        c = s["poly"].mean(axis=0)
        dup = False
        if len(keep):
            d = np.hypot(centers[:, 0] - c[0], centers[:, 1] - c[1])
            for j in np.where(d < 1.3 * max(s["l"], s["w"]))[0]:
                if d[j] < 0.55 * min(s["w"], keep[j]["w"]) or poly_iou(s["poly"], keep[j]["poly"]) > 0.35:
                    dup = True
                    break
        if not dup:
            keep.append(s)
            centers = np.vstack([centers, c])
    return keep


def row_neighbors(stalls: list[dict], exp_w: float, exp_len: float) -> list[list[int]]:
    """Melyik helyek állnak egymás mellett ugyanabban a sorban (közel azonos irány, egy beállószélességnyire, hosszában egy vonalban)."""
    if not stalls:
        return []
    C = np.array([s["poly"].mean(axis=0) for s in stalls])
    A = np.radians([s["ang"] for s in stalls])
    Dv, Nv = np.stack([np.cos(A), np.sin(A)], 1), np.stack([-np.sin(A), np.cos(A)], 1)
    out = []
    for i, s in enumerate(stalls):
        rel = C - C[i]
        along, across = np.abs(rel @ Dv[i]), np.abs(rel @ Nv[i])
        dang = np.abs((np.degrees(A) - s["ang"] + 90) % 180 - 90)
        ok = (dang <= 12) & (across > 0.55 * exp_w) & (across < 1.6 * exp_w) & (along < 0.5 * exp_len)
        ok[i] = False
        out.append(np.where(ok)[0].tolist())
    return out


def periodic_only(stalls: list[dict], exp_w: float, exp_len: float, min_row: int = 2) -> list[dict]:
    """Csíkpárokból csak a periodikus sor érvényes: legalább `min_row` egymás melletti hely, amelyek közül legalább egy
    közvetlenül két látható csík között van (nem csak pótolt)."""
    if not stalls:
        return []
    nb = row_neighbors(stalls, exp_w, exp_len)
    comp, keep = [-1] * len(stalls), []
    for i in range(len(stalls)):
        if comp[i] >= 0:
            continue
        stack, members = [i], []
        comp[i] = i
        while stack:
            x = stack.pop()
            members.append(x)
            for y in nb[x]:
                if comp[y] < 0:
                    comp[y] = i
                    stack.append(y)
        if len(members) >= min_row and sum(bool(stalls[m].get("direct")) for m in members) >= min(2, min_row):
            keep += members
    return [stalls[i] for i in sorted(keep)]


def discover(frames: list[np.ndarray], detections: list[list[dict]], transforms: list[np.ndarray],
             profile: ParkingConceptProfile | None, log: Callable[[str], None] = lambda m: None) -> dict:
    """Parkolóhelyek felfedezése egy videó mintavett képkockáiból, a 0. képkocka koordinátáiban – háromlépcsős hibrid:
      1. álló járművek sorokba rendezése, a helyük + a köztük lévő hézagok és a sorvégek üres helyei;
      2. halvány csíkpárok periodikus sorai az 1. lépcső által le nem fedett (jellemzően teljesen üres) részeken;
      3. egyesítés és duplikátumszűrés.
    Koordinátát semmi nem hoz magával: profile=None esetén (tanítás) alapértékekből indul, különben a tanult arányokkal
    és a videó saját járműméreteivel, csíkjaival dolgozik."""
    h, w = frames[0].shape[:2]
    aligned, samples = [], []
    for frame, vehicles, M in zip(frames, detections, transforms):
        inv = cv2.invertAffineTransform(M)
        aligned.append(cv2.warpAffine(frame, inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        samples.append([vehicle_from_poly(warp_points(v["poly"], inv), v["conf"], v["cls"]) for v in vehicles])
    all_v = [v for vs in samples for v in vs]
    p = profile or ParkingConceptProfile()
    notes: list[str] = []
    # háttér: sima időbeli medián – a mozgó autók eltűnnek, a parkolók a helyükön maradnak (nem takarjuk ki őket)
    bg = np.median(np.stack(aligned), axis=0).astype(np.uint8) if len(aligned) > 1 else aligned[0].copy()
    del aligned

    if len(all_v) >= 3:
        car_w, car_l = float(np.median([v["w"] for v in all_v])), float(np.median([v["l"] for v in all_v]))
    else:
        car_w = car_l = 0.0
        notes.append("A videóban nincs (elég) jármű: a helyek csak a felfestésből, a lépték a csíkok távolságából lett becsülve.")

    # ---------------- 1. lépcső: álló járművek sorai
    statics = static_vehicles(samples, car_w) if car_w > 0 else []
    rows = vehicle_rows(statics, car_w, car_l, p.step_ratio) if statics else []
    perp = [r for r in rows if not r["parallel"]]
    if perp:
        steps = [r["step"] * abs(float(r["car_dir"][0] * r["dir"][1] - r["car_dir"][1] * r["dir"][0])) for r in perp for _ in r["cars"]]
        exp_w = float(np.median(steps))                      # a helyek szélessége a csíkokra merőlegesen
        exp_len = float(np.median([r["l"] for r in perp for _ in r["cars"]])) * p.length_ratio
    elif car_w > 0:
        exp_w, exp_len = p.width_ratio * car_w, p.length_ratio * car_l
    else:
        exp_w = exp_len = 0.0
    scale = car_l / p.median_car_length_px if (p.median_car_length_px > 0 and car_l > 0) else 1.0
    thick = max(1.5, p.stripe_thickness_px * scale) if p.stripe_thickness_px > 0 else max(1.5, max(exp_w, 20.0) * p.stripe_thickness_ratio)

    # ---------------- 2. lépcső előkészítése: halvány vonalak (ezekre az 1. lépcső sorvég- és hézagvizsgálata is támaszkodik)
    if exp_w <= 0:   # nincs jármű: a párhuzamos vonalak leggyakoribb távolsága adja a helyszélességet
        guess = max(20.0, min(h, w) / 25)
        rough, mask, Lc, top = faint_lines(bg, guess, 2.0, p.stripe_delta_l)
        dists = []
        for group in angle_groups([s for s in rough if s["len"] >= 0.5 * guess], 8.0):
            nrm = unit(group[0]["ang"] + 90)
            vs = np.sort([float(nrm @ (s["p1"] + s["p2"]) / 2) for s in group])
            dists += [x for x in np.diff(vs) if x > 8]
        if len(dists) >= 3:
            hist, edges = np.histogram(dists, bins=40, range=(0, float(np.percentile(dists, 90)) * 1.2))
            exp_w = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2)
            exp_len = p.aspect_ratio * exp_w
            car_w, car_l = exp_w / p.width_ratio, exp_len / p.length_ratio
    if exp_w <= 0:
        return {"stalls": [], "bg": bg, "mask": np.zeros((h, w), np.uint8), "stripes": [], "notes": notes + ["Nem találtam sem álló járművet, sem elég csíkot."],
                "car_w": 0.0, "car_l": 0.0, "exp_w": 0.0, "exp_len": 0.0, "statics": [], "rows": [], "pairs": [], "gaps": [], "lines": [], "top": None, "Lc": None}
    lines, mask, Lc, top = faint_lines(bg, exp_len, thick, p.stripe_delta_l)
    # a jármű karosszériáján futó vonal (szélvédő, tetőél) nem felfestés: a képet nem takarjuk ki, csak ezeket a
    # vonaljelölteket hagyjuk el; a járművek közötti csíkok megmaradnak
    inside = vehicle_mask((h, w), statics, 0.86)

    def on_car(s: dict) -> bool:
        pts = np.round([(s["p1"] * (4 - k) + s["p2"] * k) / 4 for k in (1, 2, 3)]).astype(int)
        return all(0 <= x < w and 0 <= y < h and inside[y, x] > 0 for x, y in pts)

    lines = [s for s in lines if not on_car(s)]
    index = LineIndex(lines)
    # forgalom: azok az észlelések, amelyek nem álló járműhöz tartoznak (úton, sorok között haladó autók)
    sc = np.array([v["c"] for v in statics], np.float64).reshape(-1, 2)
    traffic = np.array([v["c"] for v in all_v if not len(sc) or np.min(np.hypot(sc[:, 0] - v["c"][0], sc[:, 1] - v["c"][1])) > 0.6 * car_w],
                       np.float64).reshape(-1, 2)
    # a parkolófelület jellemző fényessége és megengedett eltérése
    levels = [lv for r in rows for lv in row_surface(Lc, r, r["l"] * p.length_ratio)]
    surface = float(np.median(levels)) if len(levels) >= 3 else None
    tol = float(max(24.0, 2.5 * np.median(np.abs(np.array(levels) - surface)))) if surface is not None else 30.0

    # ---------------- 1. lépcső: helyek a sorokból
    row_slots = slots_from_rows(rows, index, p.length_ratio, (h, w), Lc, surface, tol, traffic)

    # ---------------- 2. lépcső: periodikus csíkpárok a még le nem fedett részeken
    lo, hi = 0.62 * exp_len, 1.6 * exp_len
    stripes = []
    for a in lines:   # a hosszú vonalat (gerinc, két sor közös elválasztója) a rá merőlegesek metszéspontjainál daraboljuk
        if a["len"] < lo:
            continue
        if a["len"] <= hi:
            stripes.append(a)
        elif a["len"] <= 2.6 * exp_len:
            d, mid = (a["p2"] - a["p1"]) / a["len"], a["len"] / 2
            stripes += [{"p1": a["p1"], "p2": a["p1"] + d * mid, "len": mid, "ang": a["ang"]}, {"p1": a["p1"] + d * mid, "p2": a["p2"], "len": mid, "ang": a["ang"]}]
    pair_slots, gaps = stalls_from_stripes(stripes, exp_w, exp_len, statics, lo=0.72, hi=1.45)
    for s in pair_slots:
        s["direct"] = s["src"] == "stripes"
        s["src"] = "stripes"
    asp_lo, asp_hi = p.aspect_min * 0.75, p.aspect_max * 1.3
    pair_slots = [s for s in pair_slots if asp_lo <= s["l"] / max(1.0, s["w"]) <= asp_hi and 0 <= s["poly"].mean(axis=0)[0] < w and 0 <= s["poly"].mean(axis=0)[1] < h]
    def plausible(s: dict) -> bool:
        """Csíkpárból csak üres hely jön: parkolófelület legyen, ne álljon rajta jármű, és ne járjon rajta forgalom."""
        c = s["poly"].mean(axis=0)
        for pts in (traffic, sc):
            if len(pts) and any(cv2.pointPolygonTest(s["poly"], (float(x), float(y)), False) >= 0
                                for x, y in pts[np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) < s["l"]]):
                return False
        lv = region_level(Lc, s["poly"])
        return surface is None or lv is None or abs(lv - surface) <= tol

    pair_slots = [s for s in merge_slots([], pair_slots) if plausible(s)]
    pair_slots = periodic_only(pair_slots, exp_w, exp_len, min_row=3)
    # csak ami az 1. lépcső helyeivel nem fed át (IoU < 0,30), és amin nem áll jármű (az a jármű nem sorban áll: nem parkolóhely)
    fresh = []
    rc = np.array([s["poly"].mean(axis=0) for s in row_slots]).reshape(-1, 2)
    for s in pair_slots:
        c = s["poly"].mean(axis=0)
        if len(rc):
            d = np.hypot(rc[:, 0] - c[0], rc[:, 1] - c[1])
            if any(poly_iou(s["poly"], row_slots[j]["poly"]) >= 0.30 for j in np.where(d < 1.3 * exp_len)[0]):
                continue
        fresh.append(s)
    fresh = periodic_only(fresh, exp_w, exp_len, min_row=3)

    # ---------------- 3. lépcső: egyesítés, duplikátumszűrés
    stalls = merge_slots(row_slots, fresh)
    count = {k: sum(s["src"] == k for s in stalls) for k in ("vehicle", "filled", "stripes")}
    log(f"álló jármű: {len(statics)} (mintánként kb. {len(all_v) // max(1, len(samples))} jármű), sor: {len(rows)}, vonal: {len(lines)}, "
        f"parkolóhely: {len(stalls)} (járműnél {count['vehicle']}, hézagban/sorvégen {count['filled']}, csíkpárból {count['stripes']}), "
        f"helyméret kb. {exp_w:.0f}×{exp_len:.0f} px")
    if not rows and statics:
        notes.append("Az álló járművek nem álltak össze sorokká; a helyek a felfestésből származnak.")
    return {"stalls": stalls, "bg": bg, "mask": mask, "stripes": stripes, "notes": notes, "car_w": car_w, "car_l": car_l, "exp_w": exp_w,
            "exp_len": exp_len, "statics": statics, "rows": rows, "pairs": [s for s in pair_slots if s.get("direct")], "gaps": gaps, "lines": lines, "top": top, "Lc": Lc}


def learn_profile(found: dict) -> ParkingConceptProfile:
    """A tanító videóból a parkolóhely fogalmának mérése: arányok és vizuális jellemzők, képkoordináta nélkül.
    Amire nincs elég adat, ott az alapérték marad."""
    p = ParkingConceptProfile(notes=list(found["notes"]))
    car_w, car_l = found["car_w"], found["car_l"]
    p.median_car_width_px, p.median_car_length_px = round(car_w, 2), round(car_l, 2)
    rows = [r for r in found["rows"] if r["measured"] and len(r["cars"]) >= 3]
    p.trained_stalls, p.trained_pairs, p.trained_vehicles = len(found["stalls"]), len(found["pairs"]), len(found["statics"])
    if rows:
        wts = [len(r["cars"]) for r in rows]
        # a sor lépésköze a jármű sor menti kiterjedéséhez képest: ezzel osztja be a 2. videó a hézagokat
        p.step_ratio = round(float(np.clip(np.average([r["step"] / r["ext"] for r in rows], weights=wts), 1.05, 1.6)), 3)
        perp = [r for r in rows if not r["parallel"]]
        if perp and car_w > 0:
            sw = float(np.median([r["step"] * abs(float(r["car_dir"][0] * r["dir"][1] - r["car_dir"][1] * r["dir"][0])) for r in perp]))
            p.width_ratio = round(float(np.clip(sw / car_w, 1.0, 1.75)), 3)
    else:
        p.notes.append("Kevés szomszédos álló jármű volt a tanító videóban: a lépésköz aránya alapértéken maradt.")
    pairs = found["pairs"]
    if len(pairs) >= 4 and car_l > 0:
        # a látható csíkpárokból: a beálló tényleges hossza és oldalaránya
        sl, sw = float(np.median([s["l"] for s in pairs])), float(np.median([s["w"] for s in pairs]))
        p.length_ratio = round(float(np.clip(sl / car_l, 1.0, 1.45)), 3)
        asp = np.array([s["l"] / max(1.0, s["w"]) for s in pairs])
        p.aspect_ratio = round(float(np.median(asp)), 3)
        p.aspect_min, p.aspect_max = round(float(np.percentile(asp, 5)) * 0.9, 3), round(float(np.percentile(asp, 95)) * 1.1, 3)
        if len(found["gaps"]) >= 4:
            p.stripe_gap_cv = round(float(np.std(found["gaps"]) / max(1.0, np.mean(found["gaps"]))), 3)
    mask, top = found["mask"], found["top"]
    if pairs and top is not None and mask.any():
        line = np.zeros_like(mask)
        for s in pairs:   # a csíkpárból talált helyek két hosszanti oldala a csíkokon fut
            q = np.round(s["poly"]).astype(int)
            cv2.line(line, tuple(q[0]), tuple(q[1]), 255, 1)
            cv2.line(line, tuple(q[3]), tuple(q[2]), 255, 1)
        on = (cv2.dilate(line, np.ones((3, 3), np.uint8)) > 0) & (mask > 0)
        if on.sum() > 50:
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            p.stripe_thickness_px = round(float(np.clip(2 * np.median(dist[on]), 1.5, 0.2 * max(found["exp_w"], 10))), 2)
            p.stripe_thickness_ratio = round(p.stripe_thickness_px / max(1.0, found["exp_w"]), 4)
            p.stripe_l_median = round(float(np.median(found["Lc"][on])), 1)
            p.stripe_s_p95 = round(float(np.percentile(cv2.cvtColor(found["bg"], cv2.COLOR_BGR2HSV)[:, :, 1][on], 95)), 1)
            p.stripe_delta_l = round(float(np.median(top[on])), 1)
    occ = []
    for s in found["stalls"]:
        if s["src"] != "vehicle":
            continue
        area = poly_area(s["poly"])
        best = max((intersection_area(s["poly"], v["poly"]) / max(1.0, area) for v in found["statics"]
                    if np.hypot(*(v["c"] - s["poly"].mean(axis=0))) < s["l"]), default=0.0)
        if best > 0.2:
            occ.append(best)
    if len(occ) >= 4:
        p.overlap_ratio = round(float(np.median(occ)), 3)
    if found["rows"]:
        dev = [ang_diff(v["ang"], r["ang"]) for r in found["rows"] for v in r["cars"]]
        p.vehicle_angle_p90 = round(float(np.percentile(dev, 90)), 2)
    return p


# --------------------------------------------------------------------------- #
# Foglaltság, rajzolás, videóírás
# --------------------------------------------------------------------------- #

def occupancy(stall_polys: list[np.ndarray], vehicles: list[dict]) -> list[bool]:
    """Foglalt (piros) a hely, ha bármely jármű legalább a 20%-át fedi, VAGY a jármű középpontja a helyen belül van;
    különben szabad (zöld)."""
    out = []
    centers = np.array([v["c"] for v in vehicles], np.float32).reshape(-1, 2)
    for poly in stall_polys:
        area = poly_area(poly)
        c = poly.mean(axis=0)
        reach = float(np.max(np.hypot(poly[:, 0] - c[0], poly[:, 1] - c[1]))) * 2.2
        occ = False
        for i in (np.where(np.hypot(centers[:, 0] - c[0], centers[:, 1] - c[1]) < reach)[0] if len(centers) else []):
            v = vehicles[i]
            if cv2.pointPolygonTest(poly.astype(np.float32), (float(v["c"][0]), float(v["c"][1])), False) >= 0:
                occ = True
                break
            if intersection_area(poly, v["poly"]) / max(1.0, area) >= 0.20:
                occ = True
                break
        out.append(occ)
    return out


class Smoother:
    """Hiszterézis: egy hely állapota csak akkor vált, ha egymás után többször ugyanazt látjuk (nincs villódzás)."""

    def __init__(self, n: int, need: int = 3):
        self.state: list[bool | None] = [None] * n
        self.count = [0] * n
        self.need = need

    def update(self, seen: list[bool]) -> list[bool]:
        for i, s in enumerate(seen):
            if self.state[i] is None:
                self.state[i] = s
            elif s != self.state[i]:
                self.count[i] += 1
                if self.count[i] >= self.need:
                    self.state[i], self.count[i] = s, 0
            else:
                self.count[i] = 0
        return [bool(s) for s in self.state]


FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "arial.ttf"]


def load_font(size: int):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class Hud:
    """Alsó információs sáv. A magyar ékezetek miatt a szöveget a Pillow rajzolja; az elkészült sávot szövegenként gyorsítótárazza."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.bar = max(44, int(height * 0.065))
        self.font = load_font(max(16, int(self.bar * 0.5)))
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def text_layer(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        if text not in self.cache:
            img = Image.new("L", (self.w, self.bar), 0)
            draw = ImageDraw.Draw(img)
            box = draw.textbbox((0, 0), text, font=self.font)
            draw.text(((self.w - (box[2] - box[0])) / 2 - box[0], (self.bar - (box[3] - box[1])) / 2 - box[1]), text, fill=255, font=self.font)
            alpha = np.asarray(img, np.float32)[:, :, None] / 255.0
            if len(self.cache) > 400:
                self.cache.clear()
            self.cache[text] = (alpha, 1.0 - alpha)
        return self.cache[text]

    def draw(self, frame: np.ndarray, total: int, occupied: int, free: int) -> None:
        hud_text = f"Összes: {total}      Foglalt:{occupied}     Szabad:{free}"
        roi = frame[self.h - self.bar:self.h]
        np.copyto(roi, (roi.astype(np.float32) * 0.12 + np.array([18, 18, 18], np.float32) * 0.88).astype(np.uint8))  # ~88% fedés
        roi[0, :] = (60, 60, 60)
        alpha, inv = self.text_layer(hud_text)
        np.copyto(roi, (roi.astype(np.float32) * inv + 255.0 * alpha).astype(np.uint8))


def draw_stalls(frame: np.ndarray, polys: list[np.ndarray], occupied: list[bool]) -> None:
    overlay = frame.copy()
    ipolys = [np.round(p).astype(np.int32) for p in polys]
    for p, occ in zip(ipolys, occupied):
        cv2.fillConvexPoly(overlay, p, RED if occ else GREEN)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, dst=frame)
    for p, occ in zip(ipolys, occupied):
        cv2.polylines(frame, [p], True, RED if occ else GREEN, 2, cv2.LINE_AA)


def finalize_video(raw: Path, final: Path) -> None:
    """Szabványos, mindenhol lejátszható H.264 MP4, atomi közzététellel: a végleges fájl csak teljesen kiírva jelenik meg."""
    tmp = final.with_name(final.stem + ".tmp.mp4")
    done = False
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-preset", "fast", "-crf", "23", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-an", str(tmp)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        done = r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0
    if not done:
        shutil.copyfile(raw, tmp)
    if not tmp.exists() or tmp.stat().st_size <= 0:
        raise RuntimeError("A kimeneti videó üres lett.")
    os.replace(tmp, final)
    raw.unlink(missing_ok=True)


def sample_indices(total: int, n: int) -> list[int]:
    if total <= 1:
        return [0]
    idx = {int(round(i * (total - 1) / (max(2, n) - 1))) for i in range(max(2, n))}
    return sorted(idx | {k for k in (30, 60) if k < total})   # a videó eleje sűrűbben: a 0., 30., 60. képkocka mindig benne van


def process_video(src: Path, out: Path, detector: Detector, profile: ParkingConceptProfile | None,
                  progress: Callable[..., None], log: Callable[[str], None] = lambda m: None,
                  cancelled: Callable[[], bool] = lambda: False, debug_dir: Path | None = None) -> dict:
    """Egy videó teljes feldolgozása. profile=None: tanítás (a visszaadott szótárban ott a tanult profil);
    különben elemzés a tanult fogalommal. Az eredményvideó az `out` helyre kerül."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("A videó nem nyitható meg (sérült vagy nem támogatott formátum).")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        raise RuntimeError("A videó képmérete nem olvasható ki.")

    # --- 1. menet: mintavett képkockák, járművek, sodródás -> parkolóhelyek a 0. képkocka koordinátáiban
    if total <= 0:  # egyes konténerek nem adják meg a képkockaszámot: megszámoljuk
        while cap.grab():
            total += 1
        cap.release()
        cap = cv2.VideoCapture(str(src))
    wanted = set(sample_indices(total, SAMPLES))
    frames, dets, transforms = [], [], []
    stab = None
    idx = 0
    progress(status="processing", message="Parkolóhelyek felfedezése…", total_frames=total)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cancelled():
            raise Cancelled
        if idx == 0:
            v = detector(frame)
            stab = Stabilizer(frame, v)
            frames.append(frame); dets.append(v); transforms.append(stab.M.copy())
        else:
            if idx in wanted:
                v = detector(frame)
                M = stab.update(frame, v).copy()
                frames.append(frame); dets.append(v); transforms.append(M)
            else:
                stab.update(frame)
        idx += 1
        if idx % 20 == 0:
            progress(progress=min(18, int(18 * idx / max(1, total))))
    cap.release()
    if not frames:
        raise RuntimeError("A videóból egyetlen képkocka sem olvasható ki.")
    total = idx
    found = discover(frames, dets, transforms, profile, log)
    learned = learn_profile(found) if profile is None else None
    stalls = found["stalls"]
    polys0 = [s["poly"] for s in stalls]
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "background.jpg"), found["bg"])
        cv2.imwrite(str(debug_dir / "mask.png"), found["mask"])
        dbg = found["bg"].copy()
        for s in found["lines"]:
            cv2.line(dbg, tuple(np.round(s["p1"]).astype(int)), tuple(np.round(s["p2"]).astype(int)), (255, 0, 255), 1)
        for r in found["rows"]:   # a járműsorok tengelye
            a, b = r["cars"][0]["c"], r["cars"][-1]["c"]
            cv2.line(dbg, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (0, 255, 255), 1)
        for s in stalls:
            cv2.polylines(dbg, [np.round(s["poly"]).astype(np.int32)], True, {"stripes": (0, 255, 0), "filled": (0, 200, 255), "vehicle": (255, 160, 0)}[s["src"]], 1)
        cv2.imwrite(str(debug_dir / "stalls.jpg"), dbg)
    del frames
    for key in ("bg", "mask", "Lc", "top", "lines"):
        found.pop(key, None)

    # --- 2. menet: minden képkockán foglaltság, rajzolás, kiírás
    every = DETECT_EVERY or (1 if detector.device == "cuda" else 6)
    raw = out.with_name(out.stem + ".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("A kimeneti videó nem hozható létre.")
    cap = cv2.VideoCapture(str(src))
    hud, smooth = Hud(w, h), Smoother(len(stalls), need=3 if every == 1 else 2)
    stab, state, vehicles = None, [False] * len(stalls), []
    idx = occ = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cancelled():
                raise Cancelled
            detect_now = idx % every == 0
            if detect_now:
                vehicles = detector(frame)
            if stab is None:
                stab = Stabilizer(frame, vehicles)
                M = stab.M
            else:
                M = stab.update(frame, vehicles if detect_now else None)
            polys = [warp_points(p, M) for p in polys0]
            if detect_now:
                state = smooth.update(occupancy(polys, vehicles))
            occ = int(sum(state))
            draw_stalls(frame, polys, state)
            hud.draw(frame, len(stalls), occ, len(stalls) - occ)
            writer.write(frame)
            idx += 1
            if idx % 5 == 0 or idx == total:
                progress(progress=min(97, 18 + int(79 * idx / max(1, total))), current_frame=idx, total_frames=max(total, idx),
                         occupied=occ, free=len(stalls) - occ, message="Képkockák feldolgozása…")
    finally:
        cap.release()
        writer.release()
    if idx == 0:
        raise RuntimeError("A videóból egyetlen képkocka sem olvasható ki.")
    progress(progress=98, message="Videó kódolása…")
    finalize_video(raw, out)
    return {"profile": learned, "total_slots": len(stalls), "final_occupied": occ, "final_free": len(stalls) - occ, "frames_processed": idx,
            "notes": found["notes"], "sources": {k: sum(s["src"] == k for s in stalls) for k in ("stripes", "filled", "vehicle")},
            "exp_w": found["exp_w"], "exp_len": found["exp_len"]}
