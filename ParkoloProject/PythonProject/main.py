"""
Parkolóhely-foglaltság Rendszer - Vékony Kliens Fő Indítófájl
=============================================================
Indítás:
    python main.py

Ez azonnal megnyitja a vékony kliens grafikus felhasználói felületét (GUI),
ahol lépésről lépésre (távoli AI szerveren történő feldolgozással):
1. Beállíthatod és ellenőrizheted a távoli AI szerver címét,
2. Kiválaszthatod és elküldheted a tanító videót, majd visszanézheted az eredményt,
3. Kiválaszthatod a tesztelő videót, elindíthatod a távoli elemzést,
   és ellenőrizheted a szabad/foglalt helyek jelölését.
"""

from gui import main

if __name__ == '__main__':
    main()
