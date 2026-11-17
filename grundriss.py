"""Nook-Grundriss für die lokale Maschinenplanung.

Installation: python -m pip install pillow pypdfium2 rapidocr-onnxruntime
Editor:       python grundriss.py [DATEI]
SVG-Export:   python grundriss.py DATEI --svg AUSGABE.svg [--seite 2]
Python-API:   plan = grundriss.grundriss_importieren(datei, seite=1)
              plan.maschinen.append(grundriss.Maschine(...))
              plan.speichern(datei) oder plan.svg_exportieren(datei)
"""

import argparse
import base64
import io
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from queue import Empty, SimpleQueue
from tempfile import NamedTemporaryFile
from threading import Thread
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageOps


MAX_KANTE = 2600
MAX_OBJEKTKANTE = 900
PROJEKTFORMAT = "grundriss-maschinen-v1"
PLANFARBE = "#233548"
MASCHINENRAND = "#154D8B"
MASCHINENFLAECHE = "#D5EAFF"
GRIFFE = ((-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0))


def _sicher_schreiben(pfad, inhalt):
    ziel = Path(pfad)
    with NamedTemporaryFile(dir=ziel.parent, prefix=".grundriss-", delete=False) as datei:
        temp = Path(datei.name)
        try:
            datei.write(inhalt)
            datei.close()
            temp.replace(ziel)
        finally:
            datei.close()
            temp.unlink(missing_ok=True)


@lru_cache(maxsize=32)
def _objektbild(png):
    if png:
        try:
            with Image.open(io.BytesIO(base64.b64decode(png, validate=True)), formats=["PNG"]) as bild:
                if max(bild.size) <= MAX_OBJEKTKANTE:
                    return bild.convert("RGBA")
        except (OSError, ValueError, Image.DecompressionBombError):
            pass
    return None


def _png_kodieren(bild):
    png = io.BytesIO()
    bild.save(png, format="PNG")
    return base64.b64encode(png.getvalue()).decode("ascii")


@lru_cache(maxsize=32)
def _alte_vorschau_uebernehmen(png, winkel):
    bild = _objektbild(png)
    if bild is None:
        return ""
    alpha = ImageOps.invert(ImageOps.grayscale(bild))
    objekt = Image.new("RGBA", bild.size, PLANFARBE)
    objekt.putalpha(alpha)
    if winkel:
        objekt = objekt.rotate(winkel, resample=Image.Resampling.BICUBIC, expand=True)
    zuschnitt = objekt.getbbox()
    return _png_kodieren(objekt.crop(zuschnitt)) if zuschnitt else ""


@dataclass(frozen=True)
class Maschine:
    """Zentrum (x, y) ab linker oberer Ecke, Maße in Planpixeln, Winkel im Uhrzeigersinn."""

    name: str
    x: float
    y: float
    breite: float
    tiefe: float
    winkel: float = 0
    grafik_png: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.name.strip() or len(self.name) > 80 or not self.name.isprintable():
            raise ValueError("Der Maschinenname muss 1 bis 80 lesbare Zeichen enthalten.")
        if not all(math.isfinite(wert) for wert in (self.x, self.y, self.breite, self.tiefe, self.winkel)):
            raise ValueError("Die Maschinenmaße und Koordinaten müssen gültige Zahlen sein.")
        if min(self.breite, self.tiefe) <= 0:
            raise ValueError("Breite und Tiefe müssen größer als null sein.")
        if not isinstance(self.grafik_png, str) or len(self.grafik_png) > 4600000:
            raise ValueError("Das Objektbild ist ungültig oder zu groß.")

    def _drehung(self):
        winkel = math.radians(self.winkel % 360)
        return math.cos(winkel), math.sin(winkel)

    def anker(self, u, v):
        cos, sin = self._drehung()
        b, t = self.breite / 2, self.tiefe / 2
        return self.x + u * b * cos - v * t * sin, self.y + u * b * sin + v * t * cos

    def ecken(self):
        return tuple(self.anker(u, v) for u, v in GRIFFE[::2])

    def platzieren(self, x, y, plan_groesse):
        cos, sin = map(abs, self._drehung())
        halb_b = (self.breite * cos + self.tiefe * sin) / 2
        halb_t = (self.breite * sin + self.tiefe * cos) / 2
        breite, hoehe = plan_groesse
        if 2 * halb_b > breite + 1e-8 or 2 * halb_t > hoehe + 1e-8:
            raise ValueError("Die Maschine passt mit diesen Maßen und diesem Winkel nicht auf den Plan.")
        return replace(self, x=min(max(x, halb_b), breite - halb_b),
                       y=min(max(y, halb_t), hoehe - halb_t))

    def groesse_ziehen(self, griff, dx, dy, plan_groesse, mindestmass):
        """Verschiebt die gezogene Kante und hält ihre gegenüberliegende Seite fest."""
        u, v = griff
        cos, sin = self._drehung()
        b = max(min(self.breite, mindestmass), self.breite + u * (dx * cos + dy * sin))
        t = max(min(self.tiefe, mindestmass), self.tiefe + v * (-dx * sin + dy * cos))
        quer, laengs = u * (b - self.breite) / 2, v * (t - self.tiefe) / 2
        mx, my = quer * cos - laengs * sin, quer * sin + laengs * cos
        ziel = replace(self, x=self.x + mx, y=self.y + my, breite=b, tiefe=t)
        anteil = 1.0
        for alt, neu in zip(self.ecken(), ziel.ecken()):
            for vorher, nachher, grenze in zip(alt, neu, plan_groesse):
                if nachher < 0 and nachher < vorher:
                    anteil = min(anteil, vorher / (vorher - nachher))
                elif nachher > grenze and nachher > vorher:
                    anteil = min(anteil, (grenze - vorher) / (nachher - vorher))
        anteil = max(anteil, 0)
        return replace(self, x=self.x + anteil * mx, y=self.y + anteil * my,
                       breite=self.breite + anteil * (b - self.breite),
                       tiefe=self.tiefe + anteil * (t - self.tiefe))


@dataclass(frozen=True)
class Maschinenvorlage:
    name: str
    breite: float
    tiefe: float
    winkel: float = 0
    in_metern: bool = False
    vorschau_png: str = ""
    grafik_png: str = field(default="", repr=False)

    def __post_init__(self):
        Maschine(self.name, 0, 0, self.breite, self.tiefe, self.winkel, self.grafik_png)
        if not isinstance(self.in_metern, bool) or not isinstance(self.vorschau_png, str) or len(self.vorschau_png) > 48000:
            raise ValueError("Die Maschinenablage enthält einen ungültigen Eintrag.")

    @classmethod
    def von_maschine(cls, maschine, pixel_pro_meter):
        faktor = pixel_pro_meter or 1
        return cls(maschine.name, maschine.breite / faktor, maschine.tiefe / faktor, maschine.winkel,
                   in_metern=pixel_pro_meter is not None, grafik_png=maschine.grafik_png)

    def platzieren(self, x, y, plan):
        if self.in_metern and not plan.pixel_pro_meter:
            raise ValueError("Setze den Maßstab, um diese Maschine mit ihren Maßen in Metern zu platzieren.")
        faktor = plan.pixel_pro_meter if self.in_metern else 1
        grafik = self.grafik_png or _alte_vorschau_uebernehmen(self.vorschau_png, self.winkel)
        return Maschine(self.name, x, y, self.breite * faktor, self.tiefe * faktor, self.winkel,
                        grafik).platzieren(x, y, plan.groesse)


@dataclass
class Grundriss:
    name: str
    linien: Image.Image  # L-Maske: 255 markiert Linien, 0 markiert freien Hintergrund.
    maschinen: list[Maschine] = field(default_factory=list)
    pixel_pro_meter: float | None = None
    ablage: list[Maschinenvorlage] = field(default_factory=list)
    vorlage: Image.Image | None = field(default=None, repr=False, compare=False)

    @property
    def groesse(self):
        return self.linien.size

    def _pruefen(self):
        if self.pixel_pro_meter is not None and (
            not math.isfinite(self.pixel_pro_meter) or self.pixel_pro_meter <= 0
        ):
            raise ValueError("Der gespeicherte Maßstab ist ungültig.")
        b, h = self.groesse
        for maschine in self.maschinen:
            if any(not (-1e-7 <= x <= b + 1e-7 and -1e-7 <= y <= h + 1e-7)
                   for x, y in maschine.ecken()):
                raise ValueError(f"Die Maschine „{maschine.name}“ liegt außerhalb des Plans.")

    def speichern(self, datei):
        self._pruefen()
        png = io.BytesIO()
        self.linien.convert("1", dither=Image.Dither.NONE).save(png, format="PNG")
        daten = {
            "format": PROJEKTFORMAT,
            "name": self.name,
            "linien_png": base64.b64encode(png.getvalue()).decode("ascii"),
            "pixel_pro_meter": self.pixel_pro_meter,
            "maschinen": [asdict(maschine) for maschine in self.maschinen],
            "ablage": [asdict(eintrag) for eintrag in self.ablage],
        }
        _sicher_schreiben(datei, json.dumps(daten, ensure_ascii=False, indent=2).encode("utf-8"))

    def svg_exportieren(self, datei):
        self._pruefen()
        breite, hoehe = self.groesse
        svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", viewBox=f"0 0 {breite} {hoehe}",
                         width=str(breite), height=str(hoehe))
        ET.SubElement(svg, "title").text = self.name
        ET.SubElement(svg, "rect", width=str(breite), height=str(hoehe), fill="white")
        linien = ET.SubElement(svg, "g", id="grundriss", fill=PLANFARBE, stroke=PLANFARBE,
                               attrib={"stroke-width": "0.5", "stroke-linejoin": "round"})
        # Doppelte Auflösung gibt auch einpixeligen Strichen eine geschlossene Kontur.
        maske = cv2.resize(np.asarray(self.linien), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        konturen, _ = cv2.findContours(maske, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pfade = []
        for kontur in konturen:
            punkte = cv2.approxPolyDP(kontur, 0.55, True).reshape(-1, 2)
            pfade.append("M" + " ".join(f"{x / 2 + .25:g},{y / 2 + .25:g}" for x, y in punkte) + "Z")
        ET.SubElement(linien, "path", d=" ".join(pfade), attrib={"fill-rule": "evenodd"})
        gruppe = ET.SubElement(svg, "g", id="maschinen", attrib={"font-family": "Inter, Arial, sans-serif"})
        for maschine in self.maschinen:
            eintrag = ET.SubElement(gruppe, "g")
            masse = f"{maschine.breite:.1f} × {maschine.tiefe:.1f} px"
            if self.pixel_pro_meter:
                masse = f"{maschine.breite / self.pixel_pro_meter:.2f} × {maschine.tiefe / self.pixel_pro_meter:.2f} m"
            ET.SubElement(eintrag, "title").text = f"{maschine.name} · {masse}"
            if _objektbild(maschine.grafik_png) is not None:
                ET.SubElement(eintrag, "image", x=str(-maschine.breite / 2), y=str(-maschine.tiefe / 2),
                              width=str(maschine.breite), height=str(maschine.tiefe),
                              preserveAspectRatio="none", transform=f"translate({maschine.x} {maschine.y}) rotate({maschine.winkel})",
                              href=f"data:image/png;base64,{maschine.grafik_png}")
                continue
            punkte = " ".join(f"{x:.4f},{y:.4f}" for x, y in maschine.ecken())
            ET.SubElement(eintrag, "polygon", points=punkte, fill=MASCHINENFLAECHE, stroke=MASCHINENRAND,
                          attrib={"fill-opacity": "0.9", "stroke-width": "1.5"})
            ET.SubElement(eintrag, "text", x=str(maschine.x), y=str(maschine.y), fill="#102E54",
                          attrib={"text-anchor": "middle", "dominant-baseline": "central",
                                  "font-weight": "600", "font-size": f"{_schriftgroesse(maschine, 1):g}"}).text = maschine.name
        _sicher_schreiben(datei, ET.tostring(svg, encoding="utf-8", xml_declaration=True))


def _schriftgroesse(maschine, zoom):
    platz = min(maschine.breite, maschine.tiefe) * zoom
    return max(7, min(18, platz * .78 / max(1.5, len(maschine.name) * .55)))


def _projekt_laden(pfad):
    try:
        daten = json.loads(pfad.read_text(encoding="utf-8-sig"))
        if daten["format"] != PROJEKTFORMAT:
            raise ValueError("Unbekanntes Grundriss-Projektformat.")
        with Image.open(io.BytesIO(base64.b64decode(daten["linien_png"], validate=True)), formats=["PNG"]) as png:
            if max(png.size) > MAX_KANTE:
                raise ValueError("Die Zeichenfläche des Projekts ist zu groß.")
            linien = png.convert("L").point(lambda wert: 255 if wert >= 128 else 0)
        plan = Grundriss(str(daten["name"]), linien, [Maschine(**m) for m in daten["maschinen"]],
                          daten["pixel_pro_meter"], [Maschinenvorlage(**m) for m in daten.get("ablage", [])])
        plan._pruefen()
        return plan
    except (AttributeError, KeyError, TypeError) as fehler:
        raise ValueError("Die Projektdatei enthält ungültige Grundrissdaten.") from fehler


def _pdf_oeffnen(datei):
    try:
        import pypdfium2
    except ModuleNotFoundError as fehler:
        raise ValueError("PDF-Import benötigt pypdfium2. Installation: python -m pip install pypdfium2") from fehler
    try:
        return pypdfium2.PdfDocument(datei)
    except pypdfium2.PdfiumError as fehler:
        raise ValueError(f"Die PDF-Datei lässt sich nicht öffnen. {fehler}") from fehler


def _pdf_seite(datei, nummer):
    with _pdf_oeffnen(datei) as pdf:
        if not 1 <= nummer <= len(pdf):
            raise ValueError(f"Die PDF hat {len(pdf)} Seiten. Wähle eine gültige Seitennummer.")
        seite = pdf[nummer - 1]
        try:
            kante = max(seite.get_size())
            if kante <= 0:
                raise ValueError("Die gewählte PDF-Seite ist leer.")
            bitmap = seite.render(scale=(MAX_KANTE - 1) / kante)
            try:
                return bitmap.to_pil().copy()
            finally:
                bitmap.close()
        finally:
            seite.close()


def _oeffnung(maske, breite, hoehe):
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (breite | 1, hoehe | 1))
    return cv2.morphologyEx(maske, cv2.MORPH_OPEN, kern)


def _planbild_normalisieren(grau, farben):
    # CAD-Exporte können helle Striche auf dunklem Grund enthalten.
    hintergrund = np.percentile(grau, 82)
    if hintergrund < 128:
        grau = 255 - grau
        farben = 255 - farben
        hintergrund = np.percentile(grau, 82)
    striche = np.percentile(grau, .1)
    spannweite = hintergrund - striche
    if 8 <= spannweite < 125:
        grau = np.clip((grau.astype(np.float32) - striche) * (255 / spannweite), 0, 255).astype(np.uint8)
    return grau, farben


def _scan_ausrichten(grau, farben):
    hoehe, breite = grau.shape
    faktor = min(1, 1100 / max(breite, hoehe))
    klein = cv2.resize(grau, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_AREA) if faktor < 1 else grau
    if min(klein.shape) < 25:
        return grau, farben
    kante = min(klein.shape)
    konturen = cv2.Canny(klein, 45, 125)
    segmente = cv2.HoughLinesP(konturen, 1, np.pi / 180, threshold=max(20, round(kante * .09)),
                              minLineLength=round(kante * .18), maxLineGap=max(4, round(kante * .018)))
    if segmente is None:
        return grau, farben
    segmente = segmente.reshape(-1, 4)
    dx, dy = (segmente[:, 2:] - segmente[:, :2]).T
    gewicht = np.hypot(dx, dy)
    winkel = (np.degrees(np.arctan2(dy, dx)) + 45) % 90 - 45
    sortiert = np.argsort(winkel)
    neigung = winkel[sortiert[np.searchsorted(np.cumsum(gewicht[sortiert]), gewicht.sum() / 2)]]
    zustimmung = np.abs((winkel - neigung + 45) % 90 - 45) <= 2
    waagrecht = np.abs(dx) >= np.abs(dy)
    if not (1.2 < abs(neigung) < 18 and gewicht[zustimmung].sum() >= gewicht.sum() * .70
            and np.any(zustimmung & waagrecht) and np.any(zustimmung & ~waagrecht)):
        return grau, farben
    matrix = cv2.getRotationMatrix2D((breite / 2, hoehe / 2), neigung, 1)
    cos, sin = np.abs(matrix[0, :2])
    ziel_b, ziel_h = breite * cos + hoehe * sin, hoehe * cos + breite * sin
    faktor = min(1, MAX_KANTE / max(ziel_b, ziel_h))
    matrix[:, 2] += (ziel_b - breite) / 2, (ziel_h - hoehe) / 2
    ziel = tuple(min(MAX_KANTE, math.ceil(kante * faktor)) for kante in (ziel_b, ziel_h))
    matrix *= faktor
    grau = cv2.warpAffine(grau, matrix, ziel, borderValue=255)
    farben = cv2.warpAffine(farben, matrix, ziel, borderValue=(255, 255, 255))
    return grau, farben


def _waagrechte_linien(maske, mindestlaenge, max_dicke):
    linien = _oeffnung(maske, mindestlaenge, 1)
    _, _, statistik, _ = cv2.connectedComponentsWithStats(linien, connectivity=8)
    return sorted([(int(x), int(y), int(b), int(h)) for x, y, b, h, _ in statistik[1:]
                   if b >= mindestlaenge and h <= max_dicke and b >= h * 6], key=lambda zeile: zeile[1])


def _wandfragmente_filtern(maske, kante, dicke):
    _, gruppen, statistik, _ = cv2.connectedComponentsWithStats(maske, connectivity=8)
    lang = np.maximum(statistik[:, cv2.CC_STAT_WIDTH], statistik[:, cv2.CC_STAT_HEIGHT])
    flaeche = statistik[:, cv2.CC_STAT_AREA]
    hauptwand = (lang >= kante * .10) & (flaeche >= kante * .035 * dicke)
    hauptwand[0] = False
    if not hauptwand.any():
        return maske
    kerne = hauptwand[gruppen].astype(np.uint8)
    reichweite = max(3, round(kante * .42)) | 1
    achse = max(9, round(kante * .06))
    ausgerichtet = cv2.dilate(_oeffnung(kerne, achse, 1), np.ones((1, reichweite), np.uint8))
    ausgerichtet |= cv2.dilate(_oeffnung(kerne, 1, achse), np.ones((reichweite, 1), np.uint8))
    deckung = np.bincount(gruppen[ausgerichtet > 0], minlength=len(statistik))
    behalten = hauptwand | ((deckung >= flaeche * .60) & (flaeche >= dicke ** 2))
    behalten[0] = False
    return behalten[gruppen].astype(np.uint8) * 255


def _kompakte_objekte_filtern(maske, kante):
    _, gruppen, statistik, _ = cv2.connectedComponentsWithStats(maske, connectivity=8)
    zuschnitt = statistik[1:, :4].astype(np.int64)
    if not len(zuschnitt):
        return maske
    breite, hoehe = zuschnitt[:, 2:].T
    rahmen = breite * hoehe
    # Ein massiver, kompakter Block im Inneren wird als platzierbares Objekt zur Prüfung angeboten.
    architektur = zuschnitt[np.argmax(rahmen)]
    x, y, b, h = architektur
    links, oben = zuschnitt[:, :2].T
    innen = (links > x + kante * .018) & (oben > y + kante * .018)
    innen &= (links + breite < x + b - kante * .018) & (oben + hoehe < y + h - kante * .018)
    kurz, lang = np.minimum(breite, hoehe), np.maximum(breite, hoehe)
    kompakt = (kurz >= kante * .048) & (lang <= kante * .45) & (lang <= kurz * 3.5)
    gefuellt = statistik[1:, cv2.CC_STAT_AREA] >= rahmen * .67
    behalten = np.r_[False, ~(innen & kompakt & gefuellt)]
    return behalten[gruppen].astype(np.uint8) * 255


def _einzellinien_waende(striche, laenge, dicke):
    gerade = _oeffnung(striche, laenge * 2, 1) | _oeffnung(striche, 1, laenge * 2)
    _, gruppen, statistik, _ = cv2.connectedComponentsWithStats(gerade, connectivity=8)
    flaeche = statistik[:, cv2.CC_STAT_WIDTH].astype(np.int64) * statistik[:, cv2.CC_STAT_HEIGHT]
    flaeche[0] = 0
    haupt = int(flaeche.argmax())
    b, h = statistik[haupt, 2:4]
    if not haupt or min(b, h) < laenge * 3:
        return np.zeros_like(striche)
    kerne = flaeche >= flaeche[haupt] * .22
    kerne[0] = False
    abstand = max(5, dicke * 8) | 1
    nah = cv2.dilate(kerne[gruppen].astype(np.uint8), np.ones((abstand, abstand), np.uint8))
    anschluss = np.bincount(gruppen[nah > 0], minlength=len(statistik)) > 0
    schlank = np.minimum(statistik[:, 2], statistik[:, 3]) <= dicke
    behalten = kerne | (anschluss & schlank)
    behalten[0] = False
    return behalten[gruppen].astype(np.uint8) * 255


def _waende_erkennen(grau, schwelle):
    kante = min(grau.shape)
    laenge = max(13, round(kante * .028)) | 1
    dicke = max(3, round(kante * .0044)) | 1
    _, dunkel = cv2.threshold(grau, min(142, max(64, schwelle * .65)), 255, cv2.THRESH_BINARY_INV)
    voll = _oeffnung(dunkel, laenge, dicke) | _oeffnung(dunkel, dicke, laenge)
    # Runde Öffnung erhält auch massive Wände außerhalb der horizontalen und vertikalen Achsen.
    querkern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dicke + 2, dicke + 2))
    quer = cv2.morphologyEx(dunkel, cv2.MORPH_OPEN, querkern)
    _, ids, statistik, _ = cv2.connectedComponentsWithStats(quer, connectivity=8)
    querwand = np.maximum(statistik[:, 2], statistik[:, 3]) >= laenge * 2
    querwand[0] = False
    voll |= querwand[ids].astype(np.uint8) * 255
    voll = _kompakte_objekte_filtern(voll, kante)
    massiv = np.count_nonzero(voll) / voll.size >= .009
    if massiv:
        voll = _wandfragmente_filtern(voll, kante, dicke)

    # Schraffierte Wände und CAD-Doppelkonturen liegen zwischen zwei dicht benachbarten Linien.
    _, striche = cv2.threshold(grau, min(215, max(125, schwelle + 13)), 255, cv2.THRESH_BINARY_INV)
    paare = np.zeros_like(voll)
    konturen = np.zeros_like(voll)
    for hochkant in (False, True):
        vorlage, ziel, umrandung = (striche.T, paare.T, konturen.T) if hochkant else (striche, paare, konturen)
        nah = None
        if massiv:
            querwaende = _oeffnung(voll.T if hochkant else voll, 1, laenge * 2)
            abstand = cv2.distanceTransform(cv2.bitwise_not(querwaende), cv2.DIST_L2, 3)
            nah = abstand <= max(dicke * 2, kante * .013)
        integral = cv2.integral((vorlage > 0).astype(np.uint8))
        linien = _waagrechte_linien(vorlage, laenge, dicke)
        for index, (x, y, b, h) in enumerate(linien):
            oben = y + h
            for xx, yy, bb, hh in linien[index + 1:]:
                abstand = yy - oben
                if abstand > kante * .026:
                    break
                if abstand < dicke * .65:
                    continue
                links, rechts = max(x, xx), min(x + b, xx + bb)
                strecke = rechts - links
                if strecke < max(laenge, min(b, bb) * .68, abstand * 4):
                    continue
                if nah is not None:
                    # Ergänzt dünne Trennwände an massiven Querwänden. Schmale Türflügel bleiben ausgespart.
                    if yy + hh - y < kante * .009 or strecke < kante * .125:
                        continue
                    if not nah[y:yy + hh, [links, rechts - 1]].any():
                        continue
                summe = int(integral[yy, rechts]) - int(integral[oben, rechts])
                summe -= int(integral[yy, links]) - int(integral[oben, links])
                dichte = summe / (strecke * abstand)
                if .085 <= dichte <= .75:
                    ziel[y:yy + hh, links:rechts] = 255
                elif dichte < .065 and strecke >= max(kante * .07, abstand * 8):
                    umrandung[y:yy + hh, links:rechts] = 255
    if massiv:
        return voll | (striche & (paare | konturen))
    if np.count_nonzero(paare) >= paare.size * .018:
        nah = cv2.dilate(voll | paare, np.ones((dicke * 5, dicke * 5), np.uint8))
        konturen &= nah
    kerne = _wandfragmente_filtern(voll | paare | konturen, kante, dicke)
    if not kerne.any():
        return _einzellinien_waende(striche, laenge, dicke)
    # Schwächere Strichabschnitte übernehmen, wenn sie zusammenhängend an eine erkannte Wand anschließen.
    kontrast = 255 - grau
    dichte_breite = max(5, round(kante * .009)) | 1
    dicht = np.zeros_like(voll)
    for vorlage, ziel in ((kontrast, dicht), (kontrast.T, dicht.T)):
        mittel = cv2.blur(vorlage, (laenge, dichte_breite))
        ziel |= _oeffnung((mittel >= 35).astype(np.uint8) * 255, laenge, 1)
    _, gruppen, statistik, _ = cv2.connectedComponentsWithStats(dicht, connectivity=8)
    deckung = np.bincount(gruppen[kerne > 0], minlength=len(statistik))
    anschluss = deckung >= np.maximum(8, statistik[:, cv2.CC_STAT_AREA] * .18)
    anschluss[0] = False
    region = (kerne > 0) | anschluss[gruppen]
    return (region & (kontrast >= 58)).astype(np.uint8) * 255


def _feine_striche(grau):
    block = max(13, round(min(grau.shape) * .022)) | 1
    return cv2.adaptiveThreshold(grau, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)


def _farbige_wandflaechen(farben):
    """Erkennt eine Wandfarbe anhand ihrer Verteilung entlang des Gebäudeumrisses."""
    hoehe, breite = farben.shape[:2]
    schritt = max(1, math.ceil(max(hoehe, breite) / 900))
    probe = farben[::schritt, ::schritt]
    r, g, b = (probe.astype(np.int32) // 8).transpose(2, 0, 1)
    nummern = (r * 32 + g) * 32 + b
    vorkommen = np.bincount(nummern.ravel(), minlength=32 ** 3)
    bild = farben.astype(np.int16)
    helligkeit = bild.sum(axis=2, dtype=np.int16)
    farbanteil = 3 * bild - helligkeit[:, :, None]
    mindestbreite = max(3, round(min(hoehe, breite) * .011))
    beste = None
    for nummer in np.argsort(vorkommen)[-12:][::-1]:
        anzahl = vorkommen[nummer]
        if not max(28, nummern.size * .003) <= anzahl <= nummern.size * .44:
            continue
        farbe = np.rint(np.median(probe[nummern == nummer], axis=0)).astype(np.int16)
        if np.ptp(farbe) < 8 or not 34 <= farbe.mean() <= 238:
            continue
        ton = (3 * farbe - farbe.sum()).astype(np.int32)
        # Antialiasing und JPEG können die Sättigung ändern. Neutrale Möbel- und Textstriche bleiben draußen.
        projektion = np.einsum("hwc,c->hw", farbanteil, ton, dtype=np.int32)
        staerke = projektion.astype(np.float32) / np.float32(ton @ ton)
        farbabstand = np.abs(farbanteil - staerke[:, :, None] * ton.astype(np.float32)).max(axis=2)
        nah = (staerke >= .43) & (staerke <= 1.52) & (farbabstand <= 11) & (np.abs(helligkeit - farbe.sum()) <= 72)
        _, ids, statistik, _ = cv2.connectedComponentsWithStats(nah.astype(np.uint8), connectivity=8)
        b, h, anzahl = statistik[:, 2:].T
        behalten = (np.maximum(b, h) >= mindestbreite * 2) & (anzahl >= mindestbreite * 3)
        behalten[0] = False
        maske = behalten[ids].astype(np.uint8) * 255
        x, y, breite, hoehe = cv2.boundingRect(maske)
        if min(breite, hoehe) < max(24, mindestbreite * 6):
            continue
        flaeche = np.count_nonzero(maske)
        if not .025 <= flaeche / (breite * hoehe) <= .41:
            continue
        rand = max(mindestbreite * 3, round(min(breite, hoehe) * .07))
        grundriss = maske[y:y + hoehe, x:x + breite]
        seiten = (np.any(grundriss[:rand], axis=0).mean(), np.any(grundriss[-rand:], axis=0).mean(),
                  np.any(grundriss[:, :rand], axis=1).mean(), np.any(grundriss[:, -rand:], axis=1).mean())
        if min(seiten) < .35:
            continue
        gerade = _oeffnung(maske, rand, 1) | _oeffnung(maske, 1, rand)
        anteil = np.count_nonzero(gerade) / flaeche
        if anteil < .83:
            continue
        score = min(seiten) + .30 * np.mean(seiten) + .10 * anteil
        if beste is None or score > beste[0]:
            beste = score, maske
    return beste[1] if beste is not None else None


def _doppelkonturen_ergaenzen(waende, grau, mass):
    """Ergänzt schmale Wandflächen, deren Farbe beim Verkleinern verloren ging."""
    kanten = cv2.inRange(grau, 0, 195)
    mindestlaenge = max(9, round(mass * .081))
    kandidaten = []
    for hochkant in (False, True):
        quelle = kanten.T if hochkant else kanten
        helligkeit = grau.T if hochkant else grau
        geschlossen = cv2.morphologyEx(quelle, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
        achsen = _waagrechte_linien(geschlossen, mindestlaenge, max(2, round(mass * .012)))
        for index, (x, y, b, h) in enumerate(achsen):
            for xx, yy, bb, hh in achsen[index + 1:]:
                abstand = yy - (y + h)
                if abstand > max(3, mass * .025):
                    break
                if abstand < 1:
                    continue
                a, z = max(x, xx), min(x + b, xx + bb)
                if z - a < max(mindestlaenge, min(b, bb) * .66, abstand * 4):
                    continue
                innen = helligkeit[y + h:yy, a:z]
                oben = helligkeit[max(0, y - 3):y, a:z]
                unten = helligkeit[yy + hh:yy + hh + 3, a:z]
                if oben.size and unten.size and min(np.median(oben), np.median(unten)) - np.median(innen) >= 13:
                    kandidaten.append((hochkant, a, z, y, yy + hh))

    ergaenzt = waende.copy()
    # Der zweite Durchlauf verbindet kurze Querwände mit neu bestätigten Längswänden.
    for _ in range(2):
        abstand = cv2.distanceTransform(cv2.bitwise_not(ergaenzt), cv2.DIST_L2, 3)
        for hochkant, a, z, y, yy in kandidaten:
            abstaende = abstand.T if hochkant else abstand
            toleranz = max(3, mass * .013)
            rand = min((z - a) // 4, max(5, round(mass * .035)))
            anfang = abstaende[y:yy, a:a + rand].min() <= toleranz
            ende = abstaende[y:yy, z - rand:z].min() <= toleranz
            anschluss = (anfang and ende) or (z - a >= mass * .105 and (anfang or ende))
            if anschluss and np.median(abstaende[y:yy, a:z]) >= 1.7:
                ziel = ergaenzt.T if hochkant else ergaenzt
                ziel[y:yy, a:z] = 255
    return ergaenzt


def _wandluecken(waende, mass):
    """Findet Öffnungen, deren beide Seiten über mehrere Rasterzeilen Wandkontakt haben."""
    for hochkant in (False, True):
        vorlage = waende.T if hochkant else waende
        gruppen = defaultdict(list)
        for y, zeile in enumerate(vorlage):
            kanten = np.flatnonzero(np.diff(zeile.astype(np.int16), prepend=0, append=0))
            for l, a, z, r in zip(kanten[::2], kanten[1::2], kanten[2::2], kanten[3::2]):
                b = z - a
                if max(8, round(mass * .026)) <= b <= mass * .30 and min(a - l, r - z) >= max(2, b * .068):
                    gruppen[round(a / 3), round(z / 3)].append((y, a, z))
        for zeilen in gruppen.values():
            reihen = np.array(zeilen)
            for kette in np.split(reihen, np.flatnonzero(np.diff(reihen[:, 0]) > 2) + 1):
                if len(kette) < 2:
                    continue
                y0, y1 = int(kette[0, 0]), int(kette[-1, 0])
                a, z = np.rint(np.median(kette[:, 1:], axis=0)).astype(int)
                if y1 - y0 + 1 <= max(5, (z - a) * .47):
                    yield hochkant, a, z, y0, y1


def _farbplan_oeffnungen(waende, striche):
    """Übernimmt Fensterprofile und an Wandöffnungen belegte Türblätter und Bögen aus der Vorlage."""
    hoehe, breite = waende.shape
    x, y, b, h = cv2.boundingRect(waende)
    mass = min(b, h)
    inkabstand = cv2.distanceTransform(cv2.bitwise_not(striche), cv2.DIST_L2, 3)
    wandabstand = cv2.distanceTransform(cv2.bitwise_not(waende), cv2.DIST_L2, 3)

    def ablesen(daten, punkte, hochkant):
        xx, yy = np.moveaxis(np.rint(punkte).astype(np.int32), -1, 0)
        if hochkant:
            xx, yy = yy, xx
        innen = (xx >= 0) & (xx < breite) & (yy >= 0) & (yy < hoehe)
        return np.where(innen, daten[np.clip(yy, 0, hoehe - 1), np.clip(xx, 0, breite - 1)], np.inf)

    fenster = np.zeros_like(waende)
    treffer = []
    positionen = np.linspace(.12, 1, 15)[None, :, None]
    winkel = np.radians(np.arange(5, 93, 4))[None]
    for hochkant, a, z, y0, y1 in _wandluecken(waende, mass):
        spannweite = z - a
        rand, quer = (x, b) if hochkant else (y, h)
        if y0 - rand <= quer * .05 or rand + quer - y1 <= quer * .05:
            quelle, ziel = (striche.T, fenster.T) if hochkant else (striche, fenster)
            oben = max(0, y0 - 3)
            ausschnitt = quelle[oben:y1 + 4, a:z]
            geschlossen = cv2.morphologyEx(ausschnitt.copy(), cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
            profile = np.flatnonzero(np.count_nonzero(geschlossen, axis=1) >= spannweite * .80)
            if profile.size >= 2 and profile[-1] - profile[0] >= 2:
                ziel[oben + profile, a:z] = ausschnitt[profile]

        # Breite Öffnungen können eine Tür neben einem festen Fensterfeld enthalten.
        min_rad = round(spannweite * (.44 if spannweite >= mass * .175 else .83))
        radien = np.arange(max(8, min_rad), spannweite + 4, max(1, round(spannweite * .023)))
        for seit in (-1, 1):
            for nach_innen in (-1, 1):
                scharnier_x = a if seit == 1 else z - 1
                scharnier_y = y0 if nach_innen == -1 else y1
                mitte = np.array((scharnier_x, scharnier_y))
                toleranz = max(1.45, spannweite * .026)
                versatz = max(3, round(spannweite * .045))
                for dx in range(-versatz, versatz + 1):
                    for dy in (-1, 0, 1, 2):
                        scharnier = mitte + (dx, dy * nach_innen)
                        if ablesen(wandabstand, scharnier, hochkant) > toleranz * 2.1:
                            continue
                        richtung = np.array((0, nach_innen))
                        blatt = scharnier + radien[:, None, None] * richtung * positionen
                        blattabstand = ablesen(inkabstand, blatt, hochkant)
                        sichtbar = (np.mean(blattabstand <= toleranz, axis=1) >= .80) & (blattabstand[:, -1] <= toleranz)
                        if not sichtbar.any():
                            continue
                        radius = radien[sichtbar, None]
                        bogen = scharnier + np.stack((seit * radius * np.cos(winkel), nach_innen * radius * np.sin(winkel)), axis=-1)
                        bogenabstand = ablesen(inkabstand, bogen, hochkant)
                        abdeckung = np.mean(bogenabstand <= toleranz, axis=1)
                        geschlossen = scharnier + radius * (seit, 0)
                        anschluss = ablesen(inkabstand, geschlossen[:, None] + np.array(((0, -2), (0, 0), (0, 2)))[None], hochkant)
                        abweichung = np.abs(radius[:, 0] - spannweite)
                        score = abdeckung - (.27 * bogenabstand.mean(axis=1) + .16 * blattabstand[sichtbar].mean(axis=1)) / toleranz
                        score -= .12 * np.minimum(1, abweichung / spannweite)
                        treffend = (abdeckung >= .73) & (score >= .68)
                        treffend &= (abweichung <= 3) | (np.mean(anschluss <= toleranz, axis=1) >= .66)
                        for index in np.flatnonzero(treffend):
                            fluegel = np.array((scharnier, scharnier + richtung * radius[index, 0]))
                            kontur = np.vstack((geschlossen[index], bogen[index], fluegel[-1]))
                            if hochkant:
                                fluegel, kontur = fluegel[:, ::-1], kontur[:, ::-1]
                            treffer.append((score[index], radius[index, 0], fluegel, kontur))

    tueren = np.zeros_like(waende)
    gesetzt = []
    for _, radius, fluegel, kontur in sorted(treffer, key=lambda kandidat: -kandidat[0]):
        p = fluegel[0]
        if any(np.linalg.norm(p - start) < min(radius, r) * .34 or np.linalg.norm(p - ende) < min(radius, r) * .23
               for start, ende, r in gesetzt):
            continue
        gesetzt.append((p, kontur[0], radius))
        dicke = max(3, round(radius * .052)) | 1
        cv2.polylines(tueren, [np.rint(fluegel).astype(np.int32), np.rint(kontur).astype(np.int32)], False, 255, dicke)
    return (tueren & striche) | fenster


def _randwaende_im_scan(grau, waende, striche):
    """Erhält Schraffur am Rand abgeschnittener, kleiner Bildvorlagen."""
    kante = min(grau.shape)
    dunkel = cv2.inRange(grau, 0, 156)
    geschlossen = cv2.morphologyEx(dunkel, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    laenge = max(15, round(kante * .047))
    schraffur = _oeffnung(geschlossen, laenge, 5) | _oeffnung(geschlossen, 5, laenge)
    moeglich = waende | (schraffur & striche)
    verbunden = cv2.dilate(moeglich, np.ones((3, 3), np.uint8))
    _, ids, statistik, _ = cv2.connectedComponentsWithStats(verbunden, connectivity=8)
    x, y, b, h, _ = statistik.T
    rand = max(3, round(kante * .015))
    hoehe, breite = grau.shape
    am_rand = (x <= rand) | (y <= rand) | (x + b >= breite - rand) | (y + h >= hoehe - rand)
    lang = np.maximum(b, h)
    randwand = am_rand & (lang >= kante * .11)
    randwand[0] = False
    if np.count_nonzero(moeglich[randwand[ids]]) < np.count_nonzero(waende) * .50:
        return waende
    behalten = randwand | (lang >= kante * .67)
    behalten[0] = False
    return moeglich & (behalten[ids].astype(np.uint8) * 255)


def _duenne_trennwaende(striche, waende):
    kante = min(striche.shape)
    mindestlaenge = max(12, round(kante * .078))
    luecke = max(5, round(kante * .014)) | 1
    ausgang = striche & cv2.bitwise_not(cv2.dilate(waende, np.ones((3, 3), np.uint8)))
    gerade = np.zeros_like(waende)
    segmente = []
    for hochkant in (False, True):
        vorlage = ausgang.T if hochkant else ausgang
        ziel = gerade.T if hochkant else gerade
        geschlossen = cv2.morphologyEx(vorlage, cv2.MORPH_CLOSE, np.ones((1, luecke), np.uint8))
        for x, y, b, h in _waagrechte_linien(geschlossen, mindestlaenge, max(2, round(kante * .011))):
            segmente.append((hochkant, x, y, b, h))
            ziel[y:y + h, x:x + b] = geschlossen[y:y + h, x:x + b]

    netz = cv2.dilate(gerade, np.ones((3, 3), np.uint8))
    _, ids, statistik, _ = cv2.connectedComponentsWithStats(netz, connectivity=8)
    nah = cv2.dilate(waende, np.ones((9, 9), np.uint8))
    kontakt = np.bincount(ids[nah > 0], minlength=len(statistik))
    b, h, anzahl = statistik[:, 2:].T
    wandnetz = (kontakt >= 3) & (np.minimum(b, h) >= mindestlaenge * .75)
    wandnetz &= (np.maximum(b, h) >= kante * .23) & (anzahl <= b.astype(np.int64) * h * .19)
    wandnetz[0] = False
    lang = np.zeros_like(waende)
    for hochkant, x, y, b, h in segmente:
        gruppen = ids.T if hochkant else ids
        if b >= kante * .35 and wandnetz[gruppen[y + h // 2, x + b // 2]]:
            ziel, quelle = (lang.T, gerade.T) if hochkant else (lang, gerade)
            ziel[y:y + h, x:x + b] = quelle[y:y + h, x:x + b]

    # Kürzere Striche sind Trennwände, wenn beide Enden an einen vorhandenen Wandverlauf anschließen.
    anschluss = cv2.distanceTransform(cv2.bitwise_not(waende | lang), cv2.DIST_L2, 3)
    zusaetzlich = lang.copy()
    grenze = max(5, kante * .008)
    for hochkant, x, y, b, h in segmente:
        abstand = anschluss.T if hochkant else anschluss
        zeilen = slice(y, y + h)
        profil = abstand[zeilen, x:x + b].min(axis=0)
        rand = min(b // 4, max(5, round(kante * .038)))
        links = np.flatnonzero(profil[:rand] <= grenze)
        rechts = np.flatnonzero(profil[-rand:] <= grenze)
        if links.size and rechts.size and np.median(profil) >= max(2.5, grenze * .58):
            ziel, quelle = (zusaetzlich.T, gerade.T) if hochkant else (zusaetzlich, gerade)
            strecke = slice(x + links[0], x + b - rand + rechts[-1] + 1)
            ziel[zeilen, strecke] = quelle[zeilen, strecke]
    return striche & zusaetzlich


def _tueren_erkennen(grau, striche, waende):
    """Sucht Türblätter und Schwenkbögen mit belegten Anschlüssen an beiden Seiten der Öffnung."""
    ergebnis = np.zeros_like(waende)
    segmente = cv2.createLineSegmentDetector().detect(grau)[0]
    if segmente is None:
        return ergebnis
    hoehe, breite = grau.shape
    kante = min(grau.shape)
    klein, gross = max(8, round(kante * .022)), max(12, round(kante * .18))
    geschlossen = cv2.morphologyEx(waende, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    kern = cv2.morphologyEx(geschlossen, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    kern |= _oeffnung(waende, klein, 1) | _oeffnung(waende, 1, klein)
    wandabstand = cv2.distanceTransform(cv2.bitwise_not(kern), cv2.DIST_L2, 3)
    strichabstand = cv2.distanceTransform(cv2.bitwise_not(striche), cv2.DIST_L2, 3)
    dunkle_striche = cv2.distanceTransform(cv2.inRange(grau, 176, 255), cv2.DIST_L2, 3)

    def ablesen(abstand, punkte):
        x, y = np.moveaxis(np.rint(punkte).astype(np.int32), -1, 0)
        innen = (x >= 0) & (x < breite) & (y >= 0) & (y < hoehe)
        return np.where(innen, abstand[np.clip(y, 0, hoehe - 1), np.clip(x, 0, breite - 1)], np.inf)

    winkel = np.radians(np.arange(0, 109, 3))
    blatt_proben = np.linspace(.12, 1, 14)[:, None]
    luecken_proben = np.array((.25, .42, .59, .75))[:, None]
    laibungs_proben = np.array((.08, .18, .26))[:, None]
    faktoren = np.array((.90, .95, 1, 1.05, 1.10, 1.16, 1.29, 1.44, 1.60, 1.78))
    verschiebungen = np.repeat((0, .062, -.08), len(faktoren))
    faktoren = np.tile(faktoren, 3)
    treffer = []
    for start, ende in segmente.reshape(-1, 2, 2):
        laenge = np.linalg.norm(ende - start)
        if not klein * .80 <= laenge <= gross:
            continue
        wand_start, wand_ende = ablesen(wandabstand, np.stack((start, ende)))
        for a, b, wand_a, wand_b in ((start, ende, wand_start, wand_ende), (ende, start, wand_ende, wand_start)):
            if wand_a > max(3, laenge * .15) or wand_b < max(2, laenge * .13):
                continue
            richtung = (b - a) / laenge
            quer = np.array((-richtung[1], richtung[0]))
            strahl = a + laenge * richtung * blatt_proben[4:10]
            versatz = min((0, -.9, .9), key=lambda offset: ablesen(grau, strahl + quer * offset).mean())
            ausgangspunkt = a + quer * versatz
            startwinkel = np.arctan2(richtung[1], richtung[0])
            radien = laenge * faktoren
            toleranzen = np.maximum(1.35, radien * .041)
            mitten = ausgangspunkt + np.clip(radien * verschiebungen, -2.4, 1.7)[:, None] * richtung
            fluegelspitzen = mitten + radien[:, None] * richtung
            lagerabstand = ablesen(wandabstand, mitten)
            freiraum = ablesen(wandabstand, fluegelspitzen)
            blattspuren = ablesen(strichabstand, mitten[:, None] + radien[:, None, None] * richtung * blatt_proben)
            plausibel = (radien >= klein) & (radien <= gross) & (lagerabstand <= np.maximum(2.6, radien * .14))
            plausibel &= (freiraum >= np.maximum(3, radien * .19)) & (blattspuren[:, -1] <= toleranzen)
            plausibel &= np.mean(blattspuren <= toleranzen[:, None], axis=1) >= .79
            for index in np.flatnonzero(plausibel):
                radius, mitte, fluegel = radien[index], mitten[index], fluegelspitzen[index]
                toleranz, wand_mitte = toleranzen[index], lagerabstand[index]
                blatt_mittel = blattspuren[index].mean()
                for vorzeichen in (-1, 1):
                    polar = startwinkel + winkel * vorzeichen
                    bogen = mitte + radius * np.column_stack((np.cos(polar), np.sin(polar)))
                    sichtbarkeit = ablesen(strichabstand, bogen)
                    kontrast = ablesen(dunkle_striche, bogen)
                    wandnaehe = ablesen(wandabstand, bogen)
                    for stop in range(20, 36, 2):
                        if wandnaehe[stop] > max(2, radius * .12):
                            continue
                        luecke = mitte + (bogen[stop] - mitte) * luecken_proben
                        if ablesen(wandabstand, luecke).min() < max(1.35, radius * .042):
                            continue
                        werte = sichtbarkeit[3:stop - 2]
                        deckung = np.mean(werte <= toleranz)
                        if deckung < .72 or np.median(werte) > toleranz * .66:
                            continue
                        if np.mean(kontrast[3:stop - 2] <= toleranz) < .48:
                            continue
                        if np.mean(wandnaehe[4:stop - 3] > toleranz * 1.45) < .64:
                            continue
                        # Jenseits der freien Öffnung muss auf beiden Seiten wieder Wandmaterial liegen.
                        kante_der_oeffnung = bogen[stop] - mitte
                        anschluesse = np.stack((mitte - kante_der_oeffnung * laibungs_proben,
                                               bogen[stop] + kante_der_oeffnung * laibungs_proben))
                        if np.any(ablesen(wandabstand, anschluesse).min(axis=1) > max(2, radius * .048)):
                            continue
                        score = deckung - (.18 * werte.mean() + .11 * blatt_mittel
                                           + .06 * (wandnaehe[stop] + wand_mitte)) / toleranz
                        if score >= .71:
                            treffer.append((score + stop * .0015, mitte, fluegel, radius, bogen[:stop + 1]))

    gesetzt = []
    for _, mitte, fluegel, radius, bogen in sorted(treffer, key=lambda kandidat: -kandidat[0]):
        if any(np.linalg.norm(mitte - m) < min(radius, r) * .30 or np.linalg.norm(mitte - e) < min(radius, r) * .17
               for m, e, r in gesetzt):
            continue
        gesetzt.append((mitte, bogen[-1], radius))
        dicke = max(3, round(radius * .052)) | 1
        cv2.line(ergebnis, tuple(np.rint(mitte).astype(np.int32)), tuple(np.rint(fluegel).astype(np.int32)), 255, dicke)
        cv2.polylines(ergebnis, [np.rint(bogen).astype(np.int32)], False, 255, dicke)
    return striche & ergebnis


_ZAHL = r"[+\-−±]?(?:[øØ⌀]\s*)?\d+(?:[.,]\d+)?(?:\s*(?:mm|cm|dm|m|['′\"″]))?"
_MASSZAHL = re.compile(rf"{_ZAHL}(?:\s*[×x]\s*{_ZAHL})?", re.I)
_BESCHRIFTUNG = re.compile(r"[^\W\d_]{2,}|^[HS]\s*[-=]?\s*\d", re.I)
_OBJEKTNAME = re.compile(
    r"CNC|Maschine|Roboter|Fräs|Fraes|Drehbank|Drehmaschine|Presse|Säge|Laser|Kompressor|Bohr"
    r"|\b(?:Lathe|Milling|Grinder|Drill Press|Saw|Work ?bench|Sofa|Tisch|Stuhl|Bett|Schrank|Desk|Chair)\b", re.I)


@lru_cache(maxsize=1)
def _texterkennung():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as fehler:
        raise ValueError("Für das Erkennen der Maßzahlen fehlt RapidOCR. Installation: python -m pip install rapidocr-onnxruntime") from fehler
    return RapidOCR(det_limit_side_len=1920, det_limit_type="max", max_side_len=2600,
                    intra_op_num_threads=3, inter_op_num_threads=1)


def _bemassung_erkennen(grau, waende, schwelle):
    # Kleine Maßziffern müssen vor der Texterkennung ausreichend groß abgebildet sein.
    faktor = max(1, 2050 / max(grau.shape))
    scan = cv2.resize(grau, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_CUBIC) if faktor > 1 else grau
    erkannt, _ = _texterkennung()(scan, text_score=.70, box_thresh=.31, unclip_ratio=1.32)
    zahlen = np.zeros_like(grau)
    beschriftung = np.zeros_like(grau)
    dimensionen = ([], [])
    objektnamen = []
    wandabstand = cv2.distanceTransform(cv2.bitwise_not(waende), cv2.DIST_L2, 3)
    for ecken, text, _ in erkannt or []:
        ecken = np.asarray(ecken) / faktor
        mitte = ecken.mean(axis=0)
        if _OBJEKTNAME.search(text):
            name = " ".join(unicodedata.normalize("NFKC", text).split())[:80]
            if name.isprintable():
                objektnamen.append((tuple(mitte), name))
        if not _MASSZAHL.fullmatch(text.strip().rstrip("-–")):
            if _BESCHRIFTUNG.search(text):
                aussen = np.round(mitte + (ecken - mitte) * 1.10).astype(np.int32)
                cv2.fillPoly(beschriftung, [aussen], 255)
            continue
        # Der Texterkenner gibt einen Rand zurück. Der engere Ausschnitt erhält die Originalziffern.
        ecken = np.round(mitte + (ecken - mitte) * .88).astype(np.int32)
        x, y, breite, hoehe = cv2.boundingRect(ecken)
        if hoehe > breite * 1.3:
            dimensionen[1].append((y, x, hoehe, breite, ecken, text))
        else:
            dimensionen[0].append((x, y, breite, hoehe, ecken, text))

    _, vorlage = cv2.threshold(grau, min(210, max(161, schwelle + 12)), 255, cv2.THRESH_BINARY_INV)
    for hochkant, boxen in enumerate(dimensionen):
        if not boxen:
            continue
        gegend, ziel = (vorlage.T, zahlen.T) if hochkant else (vorlage, zahlen)
        achsen = _waagrechte_linien(gegend, max(9, round(min(grau.shape) * .012)), 5)
        for x, y, breite, hoehe, ecken, text in boxen:
            mitte = x + breite / 2
            streifen = []
            for xx, yy, b, h in achsen:
                abstand = min(abs(yy - (y + hoehe)), abs(yy + h - y))
                if xx <= mitte <= xx + b and b >= max(breite * 1.4, hoehe * 3) and abstand <= hoehe * 1.35:
                    streifen.append((abstand, xx, yy, b, h))
            cx, cy = round(mitte), y + hoehe // 2
            abstandsfeld = wandabstand.T if hochkant else wandabstand
            nah = abstandsfeld[min(cy, abstandsfeld.shape[0] - 1), min(cx, abstandsfeld.shape[1] - 1)] <= hoehe * 3.5
            if not (nah or streifen or re.search(r"[.,a-z'′\"″]", text, re.I)):
                continue
            cv2.fillPoly(zahlen, [ecken], 255)
            if streifen:
                _, xx, yy, b, h = min(streifen)
                # Die Maßlinie und die kurzen Striche an ihren Enden bleiben sichtbar.
                stirn = max(2, round(hoehe * .35))
                ziel[max(0, yy - 1):yy + h + 1, xx:xx + b] = 255
                for ende in (xx, xx + b - 1):
                    ziel[max(0, yy - hoehe):yy + h + hoehe, max(0, ende - stirn):ende + stirn + 1] = 255
    linien = (waende | (vorlage & zahlen)) & cv2.bitwise_not(beschriftung)
    return linien, beschriftung, objektnamen


def _objekte_erkennen(grau, farben, linien, beschriftung, namen, schwelle):
    """Sammelt geschlossene entfernte Symbole als Vorschläge für die Maschinenablage."""
    kante = min(grau.shape)
    mindestmass = max(9, kante * .016)
    _, striche = cv2.threshold(grau, min(230, max(166, schwelle + 24)), 255, cv2.THRESH_BINARY_INV)
    rand = max(3, round(kante * .003)) | 1
    ausgeschlossen = cv2.dilate(linien | beschriftung, np.ones((rand, rand), np.uint8))
    objekte = striche & cv2.bitwise_not(ausgeschlossen)
    luecke = max(3, round(kante * .005)) | 1
    geschlossen = cv2.morphologyEx(objekte, cv2.MORPH_CLOSE, np.ones((luecke, luecke), np.uint8))
    konturen, _ = cv2.findContours(geschlossen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gefunden = []
    for kontur in konturen:
        (cx, cy), (breite, tiefe), winkel = cv2.minAreaRect(kontur)
        if min(breite, tiefe) < mindestmass or max(breite, tiefe) > min(breite, tiefe) * 8:
            continue
        rahmenflaeche = breite * tiefe
        if rahmenflaeche > grau.size * .18 or cv2.contourArea(kontur) < rahmenflaeche * .4:
            continue
        winkel = (winkel + 90) % 180 - 90
        if abs(winkel) > 45:
            breite, tiefe = tiefe, breite
            winkel -= math.copysign(90, winkel)
        x, y, b, h = cv2.boundingRect(kontur)
        namenswahl = [(math.dist((cx, cy), (nx, ny)), name) for (nx, ny), name in namen
                      if x - h * .15 <= nx <= x + b + h * .15 and y - h * .25 <= ny <= y + h * 1.25]
        name = min(namenswahl)[1] if namenswahl else ""
        maske = np.zeros((h, b), np.uint8)
        cv2.drawContours(maske, [kontur - np.array([x, y], np.int32)], 0, 255, cv2.FILLED)
        region = np.s_[y:y + h, x:x + b]
        if min(b, h) <= max(14, kante * .027):
            # Erst das Schließen der Zwischenräume verbindet Buchstaben. Ein kleines Symbol hat einen eigenen Umriss.
            _, _, umrisse, _ = cv2.connectedComponentsWithStats(objekte[region] & maske, connectivity=8)
            if not np.any((umrisse[1:, 2] >= b * .60) | (umrisse[1:, 3] >= h * .68)):
                continue
        rgb = farben[region]
        farbige_flaechen = ((rgb.max(axis=2) - rgb.min(axis=2)) >= 14).astype(np.uint8) * 255
        alpha = (striche[region] | farbige_flaechen) & maske & cv2.bitwise_not(ausgeschlossen[region])
        grafik = Image.fromarray(np.dstack((rgb, alpha)))
        # Das Bild im lokalen Koordinatensystem der Stellfläche speichern, damit es später mit ihr gedreht wird.
        bild_b, bild_h = math.ceil(breite) + 1, math.ceil(tiefe) + 1
        cos, sin = math.cos(math.radians(winkel)), math.sin(math.radians(winkel))
        versatz_x = cx - x - cos * bild_b / 2 + sin * bild_h / 2
        versatz_y = cy - y - sin * bild_b / 2 - cos * bild_h / 2
        grafik = grafik.transform((bild_b, bild_h), Image.Transform.AFFINE,
                                  (cos, -sin, versatz_x, sin, cos, versatz_y), Image.Resampling.BICUBIC)
        grafik.thumbnail((MAX_OBJEKTKANTE, MAX_OBJEKTKANTE), Image.Resampling.LANCZOS)
        gefunden.append((cy, cx, name, breite, tiefe, winkel, _png_kodieren(grafik)))
    gefunden.sort(key=lambda objekt: (objekt[0], objekt[1]))
    return [Maschinenvorlage(name or f"Objekt {index}", b, t, winkel, grafik_png=png)
            for index, (_, _, name, b, t, winkel, png) in enumerate(gefunden, 1)]


def grundriss_importieren(datei, seite=1):
    """Extrahiert Wände, Türen und Bemaßung aus einem Bild oder PDF oder lädt ein Projekt."""
    pfad = Path(datei)
    if pfad.suffix.lower() == ".json":
        return _projekt_laden(pfad)
    if pfad.suffix.lower() == ".pdf":
        bild = _pdf_seite(pfad, seite)
        name = f"{pfad.stem} · Seite {seite}"
    else:
        with Image.open(pfad, formats=["PNG", "JPEG", "WEBP"]) as quelle:
            bild = ImageOps.exif_transpose(quelle)
        name = pfad.stem
    bild.thumbnail((MAX_KANTE, MAX_KANTE), Image.Resampling.LANCZOS)
    rgba = bild.convert("RGBA")
    papier = Image.new("RGBA", rgba.size, "white")
    papier.alpha_composite(rgba)
    grau = np.asarray(papier.convert("L"))
    if int(grau.max()) - int(grau.min()) < 8:
        raise ValueError("Die gewählte Vorlage enthält keine erkennbaren Linien.")
    grau, farben = _scan_ausrichten(*_planbild_normalisieren(grau, np.asarray(papier.convert("RGB"))))
    schwelle, _ = cv2.threshold(grau, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    striche = _feine_striche(grau)
    farbwaende = _farbige_wandflaechen(farben)
    if farbwaende is not None:
        waende = farbwaende
        mass = min(cv2.boundingRect(waende)[2:])
        if mass < 230:
            waende = _doppelkonturen_ergaenzen(waende, grau, mass)
        bauteile = waende | _farbplan_oeffnungen(waende, striche)
    else:
        waende = _waende_erkennen(grau, schwelle)
        if max(grau.shape) < 900:
            waende = _randwaende_im_scan(grau, waende, striche)
        if not waende.any() or waende.all():
            raise ValueError("Die gewählte Vorlage enthält keine erkennbaren Wände.")
        waende |= _duenne_trennwaende(striche, waende)
        bauteile = _tueren_erkennen(grau, striche, waende)
    linien, beschriftung, namen = _bemassung_erkennen(grau, waende, schwelle)
    linien |= bauteile
    ablage = _objekte_erkennen(grau, farben, linien, beschriftung, namen, schwelle)
    return Grundriss(name, Image.fromarray(linien), ablage=ablage, vorlage=Image.fromarray(farben))


def _editor_starten(datei=None, seite=1):
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk

    from PIL import ImageTk

    class Editor:
        def __init__(self, fenster):
            self.fenster = fenster
            self.plan = self.projekt = self.auswahl = self.vormerkung = self.drag = None
            self.ladeantwort = None
            self.vormerk_index = self.ablagezug = self.maschinenvorschau = None
            self.quell_auswahl = None
            self.ablage_daten = ()
            self.ablage_bilder = []
            self.objekt_fotos = {}
            self.modus = ""
            self.referenz = []
            self.geaendert = False
            self.zoom = None
            self.faktor = 1
            self.ursprung = (0, 0)
            self.vorschau = self.foto = None
            self.foto_zeigt_vorlage = False
            self.direkt = tk.BooleanVar(value=True)
            self.status = tk.StringVar(value="Öffne einen Grundriss als Bild, PDF oder Projektdatei.")
            fenster.geometry("1130x720")
            fenster.minsize(830, 600)
            fenster.columnconfigure(0, weight=1)
            fenster.rowconfigure(1, weight=1)
            leiste = ttk.Frame(fenster, padding=(12, 10))
            leiste.grid(row=0, column=0, sticky="ew")
            self.oeffnen_knopf = ttk.Button(leiste, text="Grundriss öffnen …", command=self.oeffnen)
            self.oeffnen_knopf.pack(side="left", padx=(0, 8))
            self.plan_knopfe = []
            for titel, aktion in (("Projekt speichern …", self.speichern), ("SVG exportieren …", self.exportieren),
                                  ("Maßstab setzen", self.massstab), ("Ansicht einpassen", self.einpassen)):
                knopf = ttk.Button(leiste, text=titel, command=aktion)
                knopf.pack(side="left", padx=(0, 8))
                self.plan_knopfe.append(knopf)
            mitte = ttk.Frame(fenster, padding=(12, 0))
            mitte.grid(row=1, column=0, sticky="nsew")
            mitte.columnconfigure(0, weight=1)
            mitte.rowconfigure(0, weight=1)
            rahmen = ttk.Frame(mitte)
            rahmen.grid(row=0, column=0, sticky="nsew")
            rahmen.columnconfigure(0, weight=1)
            rahmen.rowconfigure(0, weight=1)
            self.canvas = tk.Canvas(rahmen, background="#E8EDF3", highlightthickness=0)
            self.canvas.grid(row=0, column=0, sticky="nsew")
            senkrecht = ttk.Scrollbar(rahmen, orient="vertical", command=self.canvas.yview)
            waagrecht = ttk.Scrollbar(rahmen, orient="horizontal", command=self.canvas.xview)
            senkrecht.grid(row=0, column=1, sticky="ns")
            waagrecht.grid(row=1, column=0, sticky="ew")
            self.canvas.configure(xscrollcommand=waagrecht.set, yscrollcommand=senkrecht.set)
            seitenleiste = ttk.Frame(mitte, padding=(12, 0, 0, 0))
            seitenleiste.grid(row=0, column=1, sticky="ns")
            ttk.Label(seitenleiste, text="Objekte im Plan", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
            self.direkt_knopf = ttk.Checkbutton(seitenleiste, text="Ziehen und Größe ändern", variable=self.direkt,
                                                command=self.aktion_beenden)
            self.direkt_knopf.pack(anchor="w", pady=(4, 0))
            placed = ttk.Frame(seitenleiste)
            placed.pack(fill="both", expand=True, pady=6)
            self.liste = tk.Listbox(placed, width=32, height=4, exportselection=False, activestyle="none")
            self.liste.pack(side="left", fill="both", expand=True)
            scroller = ttk.Scrollbar(placed, orient="vertical", command=self.liste.yview)
            scroller.pack(side="right", fill="y")
            self.liste.configure(yscrollcommand=scroller.set)
            self.liste.bind("<<ListboxSelect>>", self.listen_auswahl)
            self.liste.bind("<Double-Button-1>", lambda _: self.maschine_bearbeiten())
            self.details = ttk.Label(seitenleiste, wraplength=190)
            self.details.pack(fill="x", pady=(0, 8))
            self.hinzufuegen = ttk.Button(seitenleiste, text="Maschine hinzufügen …", command=self.maschine_hinzufuegen)
            self.hinzufuegen.pack(fill="x", pady=(0, 5))
            self.aendern = ttk.Button(seitenleiste, text="Maße und Name ändern …", command=self.maschine_bearbeiten)
            self.aendern.pack(fill="x", pady=(0, 5))
            aktionen = ttk.Frame(seitenleiste)
            aktionen.pack(fill="x")
            self.zur_ablage = ttk.Button(aktionen, text="In Ablage", command=self.maschine_ablegen)
            self.zur_ablage.pack(side="left", fill="x", expand=True, padx=(0, 4))
            self.loeschen = ttk.Button(aktionen, text="Entfernen", command=self.maschine_entfernen)
            self.loeschen.pack(side="left", fill="x", expand=True)
            ttk.Label(seitenleiste, text="Ablage und Vorschläge", font=("TkDefaultFont", 13, "bold")).pack(anchor="w", pady=(14, 0))
            self.ablage_hinweis = ttk.Label(seitenleiste, wraplength=200)
            self.ablage_hinweis.pack(anchor="w", pady=(2, 5))
            ablage_rahmen = ttk.Frame(seitenleiste)
            ablage_rahmen.pack(fill="both", expand=True)
            ttk.Style(fenster).configure("Maschinen.Treeview", rowheight=34)
            self.ablage = ttk.Treeview(ablage_rahmen, show="tree", selectmode="browse", height=4, style="Maschinen.Treeview")
            self.ablage.column("#0", width=190, minwidth=100)
            self.ablage.pack(side="left", fill="both", expand=True)
            scroller = ttk.Scrollbar(ablage_rahmen, orient="vertical", command=self.ablage.yview)
            scroller.pack(side="right", fill="y")
            self.ablage.configure(yscrollcommand=scroller.set)
            self.ablage.bind("<<TreeviewSelect>>", lambda _: self.ablage_knopfe_aktualisieren())
            self.ablage.bind("<ButtonPress-1>", self.ablagezug_starten)
            self.ablage.bind("<B1-Motion>", self.ablagezug_zeichnen)
            self.ablage.bind("<ButtonRelease-1>", self.ablagezug_beenden)
            self.ablage.bind("<Double-Button-1>", self.ablage_per_klick)
            self.ablage.bind("<Return>", self.ablage_per_klick)
            ablage_aktionen = ttk.Frame(seitenleiste)
            ablage_aktionen.pack(fill="x", pady=5)
            self.ablage_knopfe = []
            for titel, aktion in (("Platzieren", self.ablage_per_klick), ("Ändern", self.ablage_bearbeiten),
                                  ("Entfernen", self.ablage_entfernen)):
                knopf = ttk.Button(ablage_aktionen, text=titel, command=aktion)
                knopf.pack(side="left", fill="x", expand=True, padx=(0, 3))
                self.ablage_knopfe.append(knopf)
            self.aus_projekt = ttk.Button(seitenleiste, text="Maschinen aus Projekt …", command=self.maschinen_importieren)
            self.aus_projekt.pack(fill="x", pady=(0, 4))
            self.aus_vorlage = ttk.Button(seitenleiste, text="Objekt aus Vorlage wählen …", command=self.vorlage_waehlen)
            self.aus_vorlage.pack(fill="x", pady=(0, 4))
            self.bauteil = ttk.Button(seitenleiste, text="Bauteil aus Vorlage übernehmen …",
                                      command=lambda: self.vorlage_waehlen(bauteil=True))
            self.bauteil.pack(fill="x")
            ttk.Label(fenster, textvariable=self.status, anchor="w", padding=(12, 8)).grid(row=2, column=0, sticky="ew")
            self.canvas.bind("<Configure>", lambda _: self.zeichnen())
            self.canvas.bind("<Button-1>", self.klick)
            self.canvas.bind("<B1-Motion>", self.verschieben)
            self.canvas.bind("<ButtonRelease-1>", self.ziehen_beenden)
            self.canvas.bind("<MouseWheel>", self.rad_zoom)
            self.canvas.bind("<Button-4>", self.rad_zoom)
            self.canvas.bind("<Button-5>", self.rad_zoom)
            fenster.bind("<Escape>", lambda _: self.aktion_beenden())
            fenster.bind("<Control-o>", lambda _: self.oeffnen())
            fenster.bind("<Control-s>", lambda _: self.speichern())
            fenster.protocol("WM_DELETE_WINDOW", self.schliessen)
            self.aktualisieren()
            if datei:
                fenster.after_idle(lambda: self.laden(Path(datei), seite))

        def aktualisieren(self):
            name = self.plan.name if self.plan else "Grundriss"
            self.fenster.title(f"{'* ' if self.geaendert else ''}{name} | Maschinenplanung")
            frei = self.ladeantwort is None
            self.oeffnen_knopf.state(["!disabled" if frei else "disabled"])
            for knopf in (*self.plan_knopfe, self.hinzufuegen, self.aus_projekt, self.direkt_knopf):
                knopf.state(["!disabled" if self.plan and frei else "disabled"])
            for knopf in (self.aus_vorlage, self.bauteil):
                knopf.state(["!disabled" if self.plan and self.plan.vorlage is not None and frei else "disabled"])
            maschine = None if self.auswahl is None else self.plan.maschinen[self.auswahl]
            for knopf in (self.aendern, self.loeschen, self.zur_ablage):
                knopf.state(["!disabled" if maschine and frei else "disabled"])
            self.liste.configure(state="normal")
            self.liste.delete(0, "end")
            if self.plan:
                self.liste.insert("end", *(m.name for m in self.plan.maschinen))
            text = ""
            if maschine:
                self.liste.selection_set(self.auswahl)
                self.liste.see(self.auswahl)
                text = self.maschinenmasse(maschine)
            self.liste.configure(state="normal" if frei else "disabled")
            self.details.configure(text=text)
            self.ablage_aktualisieren()
            self.zeichnen()

        def maschinenmasse(self, maschine):
            faktor = self.plan.pixel_pro_meter or 1
            einheit = "m" if self.plan.pixel_pro_meter else "px"
            return f"{maschine.breite / faktor:.2f} × {maschine.tiefe / faktor:.2f} {einheit} · {maschine.winkel:g}°"

        def ablage_knopfe_aktualisieren(self):
            aktiv = self.ladeantwort is None and bool(self.ablage.selection())
            for knopf in self.ablage_knopfe:
                knopf.state(["!disabled" if aktiv else "disabled"])

        def ablage_aktualisieren(self):
            eintraege = tuple(self.plan.ablage) if self.plan else ()
            if eintraege != self.ablage_daten:
                self.ablage_daten = eintraege
                self.ablage.delete(*self.ablage.get_children())
                self.ablage_bilder.clear()
                for index, eintrag in enumerate(eintraege):
                    icon = Image.new("RGB", (40, 28), MASCHINENFLAECHE)
                    bild = _objektbild(eintrag.grafik_png or eintrag.vorschau_png)
                    if bild is not None:
                        mini = ImageOps.contain(bild, (40, 28), Image.Resampling.LANCZOS)
                        icon = Image.new("RGBA", (40, 28), "white")
                        icon.alpha_composite(mini, ((40 - mini.width) // 2, (28 - mini.height) // 2))
                    foto = ImageTk.PhotoImage(icon)
                    self.ablage_bilder.append(foto)
                    self.ablage.insert("", "end", iid=str(index), text=eintrag.name, image=foto)
            if not eintraege:
                hinweis = "Keine Objekte in der Ablage."
            elif self.direkt.get():
                hinweis = "Objekt in den Plan ziehen oder „Platzieren“ wählen."
            else:
                hinweis = "Objekt auswählen und „Platzieren“ wählen."
            self.ablage_hinweis.configure(text=hinweis)
            self.ablage_knopfe_aktualisieren()

        def zeichnen(self):
            self.canvas.delete("all")
            if not self.plan:
                return
            self.aktive_fotos = {}
            b, h = self.plan.groesse
            viewport_b, viewport_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            self.faktor = self.zoom or max(.001, min((viewport_b - 32) / b, (viewport_h - 32) / h))
            anzeige = max(1, round(b * self.faktor)), max(1, round(h * self.faktor))
            zeigt_vorlage = self.modus in ("objektauswahl", "bauteilauswahl")
            if (self.foto is None or anzeige != (self.foto.width(), self.foto.height())
                    or self.foto_zeigt_vorlage != zeigt_vorlage):
                quelle = self.plan.vorlage if zeigt_vorlage else self.vorschau
                self.foto = ImageTk.PhotoImage(quelle.resize(anzeige, Image.Resampling.LANCZOS))
                self.foto_zeigt_vorlage = zeigt_vorlage
            x, y = max(16, (viewport_b - anzeige[0]) / 2), max(16, (viewport_h - anzeige[1]) / 2)
            self.ursprung = x, y
            self.canvas.configure(scrollregion=(0, 0, max(viewport_b, anzeige[0] + 32), max(viewport_h, anzeige[1] + 32)))
            self.canvas.create_image(x, y, image=self.foto, anchor="nw")
            if zeigt_vorlage:
                if self.quell_auswahl:
                    x0, y0, x1, y1 = self.quell_auswahl
                    self.canvas.create_rectangle(*self.zur_ansicht(x0, y0), *self.zur_ansicht(x1, y1),
                                                 outline="#154D8B", width=2, dash=(4, 3))
                self.objekt_fotos.clear()
                return
            for index, maschine in enumerate(self.plan.maschinen):
                if self.drag and index == self.drag[0] and self.maschinenvorschau:
                    maschine = self.maschinenvorschau
                self.maschine_zeichnen(maschine, f"maschine-{index}", index == self.auswahl)
            if self.ablagezug and self.maschinenvorschau:
                self.maschine_zeichnen(self.maschinenvorschau, "platzierung", False, vorschau=True)
            if self.direkt.get() and self.auswahl is not None and not self.modus and not self.ablagezug:
                maschine = self.maschinenvorschau or self.plan.maschinen[self.auswahl]
                for index, griff in enumerate(GRIFFE):
                    x, y = self.zur_ansicht(*maschine.anker(*griff))
                    self.canvas.create_rectangle(x - 4, y - 4, x + 4, y + 4, fill="white", outline="#A93D1D",
                                                 width=2, tags=(f"griff-{index}", f"maschine-{self.auswahl}"))
            if self.referenz:
                x, y = self.zur_ansicht(*self.referenz[0])
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#C1451A", outline="white")
            self.objekt_fotos = self.aktive_fotos

        def maschine_zeichnen(self, maschine, tag, ausgewaehlt, vorschau=False):
            rand = "#C1451A" if ausgewaehlt else MASCHINENRAND
            bild = _objektbild(maschine.grafik_png)
            if bild is not None:
                b = max(1, round(maschine.breite * self.faktor))
                h = max(1, round(maschine.tiefe * self.faktor))
                winkel = maschine.winkel % 360
                schluessel = maschine.grafik_png, b, h, winkel
                foto = self.objekt_fotos.get(schluessel) or self.aktive_fotos.get(schluessel)
                if foto is None:
                    bild = bild.resize((b, h), Image.Resampling.LANCZOS)
                    if winkel:
                        bild = bild.rotate(-winkel, resample=Image.Resampling.BICUBIC, expand=True)
                    foto = ImageTk.PhotoImage(bild)
                self.aktive_fotos[schluessel] = foto
                self.canvas.create_image(*self.zur_ansicht(maschine.x, maschine.y), image=foto, tags=tag)
                if not (ausgewaehlt or vorschau):
                    return
            ecken = [wert for punkt in maschine.ecken() for wert in self.zur_ansicht(*punkt)]
            self.canvas.create_polygon(ecken, fill=MASCHINENFLAECHE if bild is None else "", outline=rand, width=2, tags=tag,
                                       dash=(4, 3) if vorschau else ())
            if bild is None:
                self.canvas.create_text(*self.zur_ansicht(maschine.x, maschine.y), text=maschine.name,
                                        font=("TkDefaultFont", -round(_schriftgroesse(maschine, self.faktor)), "bold"),
                                        fill="#102E54", tags=tag)

        def zur_ansicht(self, x, y):
            return x * self.faktor + self.ursprung[0], y * self.faktor + self.ursprung[1]

        def zum_plan(self, event):
            return ((self.canvas.canvasx(event.x) - self.ursprung[0]) / self.faktor,
                    (self.canvas.canvasy(event.y) - self.ursprung[1]) / self.faktor)

        def _planposition_am_zeiger(self, event):
            x = event.x_root - self.canvas.winfo_rootx()
            y = event.y_root - self.canvas.winfo_rooty()
            if 0 <= x < self.canvas.winfo_width() and 0 <= y < self.canvas.winfo_height():
                px = (self.canvas.canvasx(x) - self.ursprung[0]) / self.faktor
                py = (self.canvas.canvasy(y) - self.ursprung[1]) / self.faktor
                b, h = self.plan.groesse
                if 0 <= px <= b and 0 <= py <= h:
                    return px, py
            return None

        def einpassen(self):
            self.zoom = None
            self.zeichnen()
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)

        def rad_zoom(self, event):
            if not self.plan or self.drag or self.ablagezug or self.quell_auswahl:
                return
            x, y = self.zum_plan(event)
            vergroessern = event.num == 4 or event.delta > 0
            self.zoom = min(4200 / max(self.plan.groesse), max(.03, self.faktor * (1.2 if vergroessern else 1 / 1.2)))
            self.zeichnen()
            x, y = self.zur_ansicht(x, y)
            _, _, b, h = map(float, self.canvas.cget("scrollregion").split())
            self.canvas.xview_moveto((x - event.x) / b)
            self.canvas.yview_moveto((y - event.y) / h)

        def aktion_beenden(self):
            self.modus, self.vormerkung, self.drag = "", None, None
            self.vormerk_index = self.ablagezug = self.maschinenvorschau = None
            self.quell_auswahl = None
            self.referenz.clear()
            self.canvas.configure(cursor="")
            if self.plan and self.ladeantwort is None:
                skala = f"{self.plan.pixel_pro_meter:.2f} px/m" if self.plan.pixel_pro_meter else "Maße in Planpixeln"
                hinweis = "Maschinen ziehen oder an den Griffen skalieren." if self.direkt.get() else "Maschinen per Dialog ändern."
                self.status.set(f"{skala}   ·   {hinweis}")
            self.aktualisieren()

        def massstab(self):
            self.aktion_beenden()
            self.modus = "massstab"
            self.canvas.configure(cursor="crosshair")
            self.status.set("Klicke den ersten Punkt einer Strecke mit bekannter Länge an.")

        def klick(self, event):
            if not self.plan or self.ladeantwort is not None:
                return
            x, y = self.zum_plan(event)
            b, h = self.plan.groesse
            if self.modus and not (0 <= x <= b and 0 <= y <= h):
                return
            if self.modus in ("objektauswahl", "bauteilauswahl"):
                self.quell_auswahl = x, y, x, y
                return
            if self.modus == "platzieren":
                self.platzierung_bestaetigen(self.vormerkung, self.vormerk_index, x, y)
                return
            if self.modus == "massstab":
                if not self.referenz:
                    self.referenz.append((x, y))
                    self.status.set("Klicke den zweiten Punkt der Referenzstrecke an.")
                    self.zeichnen()
                    return
                distanz = math.dist((x, y), self.referenz[0])
                if not distanz:
                    self.status.set("Wähle einen zweiten Punkt mit größerem Abstand.")
                    return
                eingabe = simpledialog.askstring("Maßstab", "Wie lang ist die Strecke in Metern?",
                                                parent=self.fenster)
                if eingabe is not None:
                    try:
                        meter = float(eingabe.replace(",", "."))
                        skala = distanz / meter
                        if meter <= 0 or not math.isfinite(skala) or skala <= 0:
                            raise ValueError()
                    except (ValueError, ZeroDivisionError):
                        self.status.set("Gib eine positive Länge in Metern ein. Klicke den zweiten Punkt erneut an.")
                        return
                    self.plan.pixel_pro_meter = skala
                    self.geaendert = True
                self.aktion_beenden()
                return
            tags = self.canvas.gettags("current")
            self.auswahl = next((int(tag.removeprefix("maschine-")) for tag in tags
                                 if tag.startswith("maschine-")), None)
            if self.auswahl is not None and self.direkt.get():
                maschine = self.plan.maschinen[self.auswahl]
                griff = next((GRIFFE[int(tag.removeprefix("griff-"))] for tag in tags if tag.startswith("griff-")), None)
                self.drag = self.auswahl, maschine, x, y, griff
                self.canvas.configure(cursor="fleur" if griff is None else "crosshair")
            self.aktualisieren()

        def verschieben(self, event):
            if self.quell_auswahl:
                x0, y0, _, _ = self.quell_auswahl
                x, y = self.zum_plan(event)
                b, h = self.plan.groesse
                self.quell_auswahl = x0, y0, min(b, max(0, x)), min(h, max(0, y))
                self.zeichnen()
                return
            if self.drag is None:
                return
            _, maschine, start_x, start_y, griff = self.drag
            x, y = self.zum_plan(event)
            dx, dy = x - start_x, y - start_y
            if griff is None:
                self.maschinenvorschau = maschine.platzieren(maschine.x + dx, maschine.y + dy, self.plan.groesse)
            else:
                self.maschinenvorschau = maschine.groesse_ziehen(griff, dx, dy, self.plan.groesse, 8 / self.faktor)
            self.details.configure(text=self.maschinenmasse(self.maschinenvorschau))
            self.zeichnen()

        def ziehen_beenden(self, _event):
            if self.quell_auswahl:
                self.vorlagenauswahl_bestaetigen()
                return
            if self.drag is None:
                return
            index, vorher, *_ = self.drag
            if self.maschinenvorschau is not None and self.maschinenvorschau != vorher:
                self.plan.maschinen[index] = self.maschinenvorschau
                self.geaendert = True
            self.aktion_beenden()

        def vorlage_waehlen(self, bauteil=False):
            self.aktion_beenden()
            self.modus = "bauteilauswahl" if bauteil else "objektauswahl"
            self.canvas.configure(cursor="crosshair")
            self.status.set("Ziehe einen engen Rahmen um die Tür oder Wand. Esc bricht ab." if bauteil
                            else "Ziehe einen engen Rahmen um das Möbelstück oder die Maschine. Esc bricht ab.")
            self.zeichnen()

        def vorlagenauswahl_bestaetigen(self):
            x0, y0, x1, y1 = self.quell_auswahl
            x, xx = sorted((round(x0), round(x1)))
            y, yy = sorted((round(y0), round(y1)))
            self.quell_auswahl = None
            if min(xx - x, yy - y) < 3:
                self.zeichnen()
                return
            rgb = self.plan.vorlage.crop((x, y, xx, yy)).convert("RGBA")
            if self.modus == "bauteilauswahl":
                maske = _feine_striche(np.asarray(rgb.convert("L")))
                bisher = np.asarray(self.plan.linien.crop((x, y, xx, yy)))
                neu = bisher | maske
                if np.array_equal(neu, bisher):
                    self.status.set("In der Auswahl wurden keine zusätzlichen Striche gefunden.")
                    self.zeichnen()
                    return
                self.plan.linien.paste(Image.fromarray(neu), (x, y))
                self.vorschau = ImageOps.colorize(self.plan.linien, "white", PLANFARBE)
                self.geaendert = True
                self.aktion_beenden()
                return
            rgb.putalpha(ImageOps.grayscale(rgb).point(lambda wert: 255 if wert < 244 else 0))
            if not rgb.getbbox():
                self.status.set("Der Rahmen ist leer. Wähle das Objekt in der Vorlage aus.")
                self.zeichnen()
                return
            rgb.thumbnail((MAX_OBJEKTKANTE, MAX_OBJEKTKANTE), Image.Resampling.LANCZOS)
            index = len(self.plan.ablage)
            name = f"Objekt {index + 1}"
            self.plan.ablage.append(Maschinenvorlage(name, xx - x, yy - y, grafik_png=_png_kodieren(rgb)))
            self.geaendert = True
            self.aktion_beenden()
            self.ablage_auswaehlen(index)

        def platzierung_bestaetigen(self, vorlage, index, x, y):
            try:
                maschine = vorlage.platzieren(x, y, self.plan)
            except ValueError as fehler:
                self.aktion_beenden()
                self.status.set(str(fehler))
                return
            self.plan.maschinen.append(maschine)
            if index is not None:
                del self.plan.ablage[index]
            self.auswahl = len(self.plan.maschinen) - 1
            self.geaendert = True
            self.aktion_beenden()

        def ablage_index(self):
            auswahl = self.ablage.selection()
            return int(auswahl[0]) if auswahl and self.ladeantwort is None else None

        def ablage_auswaehlen(self, index):
            eintrag = str(index)
            self.ablage.focus(eintrag)
            self.ablage.selection_set(eintrag)
            self.ablage.see(eintrag)
            self.ablage_knopfe_aktualisieren()

        def ablage_per_klick(self, _event=None):
            index = self.ablage_index()
            if index is None:
                return
            self.aktion_beenden()
            self.vormerkung = self.plan.ablage[index]
            self.vormerk_index, self.modus = index, "platzieren"
            self.canvas.configure(cursor="crosshair")
            self.status.set(f"Klicke auf den geplanten Mittelpunkt von „{self.vormerkung.name}“.")
            self.zeichnen()

        def ablagezug_starten(self, event):
            eintrag = self.ablage.identify_row(event.y)
            if not eintrag or self.ladeantwort is not None:
                return
            self.aktion_beenden()
            self.ablage.focus_set()
            self.ablage_auswaehlen(eintrag)
            if self.direkt.get():
                self.ablagezug = int(eintrag), event.x_root, event.y_root

        def ablagezug_zeichnen(self, event):
            if self.ablagezug is None:
                return
            index, start_x, start_y = self.ablagezug
            if math.dist((event.x_root, event.y_root), (start_x, start_y)) < 5:
                return
            ziel = self._planposition_am_zeiger(event)
            self.maschinenvorschau = None
            if ziel:
                try:
                    self.maschinenvorschau = self.plan.ablage[index].platzieren(*ziel, self.plan)
                except ValueError as fehler:
                    self.status.set(str(fehler))
            self.zeichnen()

        def ablagezug_beenden(self, event):
            if self.ablagezug is None:
                return
            index, start_x, start_y = self.ablagezug
            ziel = self._planposition_am_zeiger(event)
            if ziel and math.dist((event.x_root, event.y_root), (start_x, start_y)) >= 5:
                self.platzierung_bestaetigen(self.plan.ablage[index], index, *ziel)
            else:
                self.aktion_beenden()

        def listen_auswahl(self, _):
            werte = self.liste.curselection()
            if werte:
                self.auswahl = werte[0]
                self.aktion_beenden()

        def maschine_hinzufuegen(self):
            self.aktion_beenden()
            b, h = self.plan.groesse
            kante = min(b, h)
            faktor = self.plan.pixel_pro_meter or 1
            if self.plan.pixel_pro_meter:
                breite, tiefe = min(b / 2, 2 * faktor), min(h / 2, faktor)
            else:
                breite, tiefe = kante / 10, kante / 14
            self.maschinendialog(Maschine("Maschine", b / 2, h / 2, breite, tiefe), neu=True)

        def maschine_bearbeiten(self):
            if self.auswahl is not None and self.ladeantwort is None:
                self.aktion_beenden()
                self.maschinendialog(self.plan.maschinen[self.auswahl], neu=False)

        def ablage_bearbeiten(self):
            index = self.ablage_index()
            if index is not None:
                self.aktion_beenden()
                vorlage = self.plan.ablage[index]
                maschine = Maschine(vorlage.name, 0, 0, vorlage.breite, vorlage.tiefe, vorlage.winkel)
                self.maschinendialog(maschine, neu=False, ablage_index=index)

        def maschinendialog(self, maschine, neu, ablage_index=None):
            dialog = tk.Toplevel(self.fenster)
            dialog.title("Maschine hinzufügen" if neu else "Maschine ändern")
            dialog.transient(self.fenster)
            dialog.resizable(False, False)
            inhalt = ttk.Frame(dialog, padding=16)
            inhalt.grid()
            faktor = self.plan.pixel_pro_meter or 1
            einheit = "m" if self.plan.pixel_pro_meter else "px"
            if ablage_index is not None:
                faktor = 1
                einheit = "m" if self.plan.ablage[ablage_index].in_metern else "px"
            werte = []
            for zeile, (label, wert) in enumerate((("Name", "" if neu else maschine.name),
                                                   (f"Breite ({einheit})", format(maschine.breite / faktor, ".12g")),
                                                   (f"Tiefe ({einheit})", format(maschine.tiefe / faktor, ".12g")),
                                                   ("Drehung (°)", format(maschine.winkel, "g")))):
                ttk.Label(inhalt, text=label).grid(row=zeile, column=0, sticky="w", pady=4, padx=(0, 12))
                eingabe = ttk.Entry(inhalt, width=26)
                eingabe.grid(row=zeile, column=1, sticky="ew", pady=4)
                eingabe.insert(0, wert)
                werte.append(eingabe)
            fehlertext = ttk.Label(inhalt, foreground="#A02E15", wraplength=310)
            fehlertext.grid(row=4, columnspan=2, sticky="w")

            def bestaetigen():
                try:
                    b, t, winkel = (float(e.get().strip().replace(",", ".")) for e in werte[1:])
                except ValueError:
                    fehlertext.configure(text="Gib Breite, Tiefe und Drehung als Zahlen ein.")
                    return
                try:
                    ergebnis = replace(maschine, name=werte[0].get().strip(), breite=b * faktor, tiefe=t * faktor,
                                        winkel=winkel % 360)
                    if ablage_index is None:
                        ergebnis = ergebnis.platzieren(ergebnis.x, ergebnis.y, self.plan.groesse)
                except ValueError as fehler:
                    fehlertext.configure(text=str(fehler))
                    return
                dialog.destroy()
                if ablage_index is not None:
                    vorher = self.plan.ablage[ablage_index]
                    danach = replace(vorher, name=ergebnis.name, breite=ergebnis.breite, tiefe=ergebnis.tiefe, winkel=ergebnis.winkel)
                    self.geaendert = self.geaendert or vorher != danach
                    self.plan.ablage[ablage_index] = danach
                    self.aktualisieren()
                    self.ablage_auswaehlen(ablage_index)
                elif neu:
                    self.vormerkung = Maschinenvorlage.von_maschine(ergebnis, self.plan.pixel_pro_meter)
                    self.modus = "platzieren"
                    self.canvas.configure(cursor="crosshair")
                    self.status.set(f"Klicke auf den geplanten Mittelpunkt von „{ergebnis.name}“.")
                else:
                    self.geaendert = self.geaendert or ergebnis != maschine
                    self.plan.maschinen[self.auswahl] = ergebnis
                    self.aktualisieren()

            ttk.Button(inhalt, text="Abbrechen", command=dialog.destroy).grid(row=5, column=0, sticky="w", pady=(10, 0))
            ttk.Button(inhalt, text="Position wählen" if neu else "Übernehmen", command=bestaetigen).grid(
                row=5, column=1, sticky="e", pady=(10, 0))
            dialog.bind("<Return>", lambda _: bestaetigen())
            dialog.bind("<Escape>", lambda _: dialog.destroy())
            dialog.wait_visibility()
            dialog.grab_set()
            werte[0].focus_set()

        def maschine_entfernen(self):
            if self.auswahl is not None:
                del self.plan.maschinen[self.auswahl]
                self.auswahl = None
                self.geaendert = True
                self.aktion_beenden()

        def maschine_ablegen(self):
            if self.auswahl is not None:
                maschine = self.plan.maschinen.pop(self.auswahl)
                self.plan.ablage.append(Maschinenvorlage.von_maschine(maschine, self.plan.pixel_pro_meter))
                self.auswahl = None
                self.geaendert = True
                self.aktion_beenden()
                self.ablage_auswaehlen(len(self.plan.ablage) - 1)

        def ablage_entfernen(self):
            index = self.ablage_index()
            if index is not None:
                del self.plan.ablage[index]
                self.geaendert = True
                self.aktion_beenden()

        def maschinen_importieren(self):
            datei = filedialog.askopenfilename(parent=self.fenster, title="Maschinen aus Projekt übernehmen",
                                               filetypes=[("Grundrissprojekt", "*.json")])
            if not datei:
                return
            try:
                quelle = _projekt_laden(Path(datei))
                eintraege = [Maschinenvorlage.von_maschine(m, quelle.pixel_pro_meter) for m in quelle.maschinen]
                eintraege.extend(quelle.ablage)
            except (OSError, ValueError, Image.DecompressionBombError) as fehler:
                messagebox.showerror("Maschinen konnten nicht übernommen werden", str(fehler), parent=self.fenster)
                return
            if not eintraege:
                self.status.set("Das gewählte Projekt enthält keine Maschinen oder Vorschläge.")
                return
            erster_index = len(self.plan.ablage)
            self.plan.ablage.extend(eintraege)
            self.geaendert = True
            self.aktion_beenden()
            self.ablage_auswaehlen(erster_index)
            self.status.set(f"Übernommene Einträge in der Ablage: {len(eintraege)}")

        def wechsel_erlaubt(self):
            if not self.geaendert:
                return True
            speichern = messagebox.askyesnocancel("Änderungen speichern?", "Möchtest du das Projekt vor dem Schließen speichern?",
                                                  parent=self.fenster)
            return speichern is False or (speichern is True and self.speichern())

        def schliessen(self):
            if self.ladeantwort is not None or self.wechsel_erlaubt():
                self.fenster.destroy()

        def oeffnen(self):
            if self.ladeantwort is not None:
                return
            datei = filedialog.askopenfilename(parent=self.fenster, title="Grundriss öffnen", filetypes=[
                ("Grundrisse und Projekte", "*.png *.jpg *.jpeg *.webp *.pdf *.json"), ("Alle Dateien", "*.*")])
            if datei and self.wechsel_erlaubt():
                self.laden(Path(datei))

        def laden(self, pfad, seite=None):
            try:
                if pfad.suffix.lower() == ".pdf" and seite is None:
                    with _pdf_oeffnen(pfad) as pdf:
                        anzahl = len(pdf)
                    seite = simpledialog.askinteger("PDF-Seite auswählen", f"Welche Seite möchtest du bearbeiten? (1–{anzahl})",
                                                    parent=self.fenster, minvalue=1, maxvalue=anzahl) if anzahl > 1 else 1
                    if seite is None:
                        return
            except (OSError, ValueError, Image.DecompressionBombError) as fehler:
                messagebox.showerror("Grundriss konnte nicht geöffnet werden", str(fehler), parent=self.fenster)
                return
            self.aktion_beenden()
            antwort = self.ladeantwort = SimpleQueue()
            self.status.set("Grundriss wird geöffnet …" if pfad.suffix.lower() == ".json"
                            else "Wände, Türen, Bemaßung und mögliche Maschinen werden erkannt …")
            self.aktualisieren()

            def importieren():
                try:
                    antwort.put((grundriss_importieren(pfad, seite=seite or 1), None))
                except Exception as fehler:
                    antwort.put((None, fehler))

            Thread(target=importieren, daemon=True).start()
            self.fenster.after(80, self.laden_abschliessen, pfad)

        def laden_abschliessen(self, pfad):
            try:
                plan, fehler = self.ladeantwort.get_nowait()
            except Empty:
                self.fenster.after(80, self.laden_abschliessen, pfad)
                return
            self.ladeantwort = None
            if fehler:
                self.aktualisieren()
                self.status.set("Der Grundriss konnte nicht geöffnet werden.")
                messagebox.showerror("Grundriss konnte nicht geöffnet werden", str(fehler), parent=self.fenster)
                return
            self.plan = plan
            self.projekt = pfad if pfad.suffix.lower() == ".json" else None
            self.auswahl = None
            self.geaendert = False
            self.vorschau = ImageOps.colorize(plan.linien, "white", PLANFARBE)
            self.foto = None
            self.zoom = None
            self.aktion_beenden()
            self.einpassen()

        def speichern(self):
            if not self.plan or self.ladeantwort is not None:
                return False
            datei = self.projekt or filedialog.asksaveasfilename(
                parent=self.fenster, title="Projekt speichern", initialfile=f"{self.plan.name}.grundriss.json",
                defaultextension=".json", filetypes=[("Grundrissprojekt", "*.json")])
            if not datei:
                return False
            if Path(datei).suffix.lower() != ".json":
                messagebox.showerror("Dateiname ungültig", "Projektdateien benötigen die Endung .json.", parent=self.fenster)
                return False
            try:
                self.plan.speichern(datei)
            except (OSError, ValueError) as fehler:
                messagebox.showerror("Projekt konnte nicht gespeichert werden", str(fehler), parent=self.fenster)
                return False
            self.projekt = Path(datei)
            self.geaendert = False
            self.aktualisieren()
            self.status.set(f"Projekt gespeichert · {self.projekt.name}")
            return True

        def exportieren(self):
            datei = filedialog.asksaveasfilename(parent=self.fenster, title="SVG exportieren",
                                                initialfile=f"{self.plan.name}.svg", defaultextension=".svg",
                                                filetypes=[("SVG-Zeichnung", "*.svg")])
            if datei:
                if Path(datei).suffix.lower() != ".svg":
                    messagebox.showerror("Dateiname ungültig", "SVG-Zeichnungen benötigen die Endung .svg.", parent=self.fenster)
                    return
                try:
                    self.plan.svg_exportieren(datei)
                except (OSError, ValueError) as fehler:
                    messagebox.showerror("SVG konnte nicht exportiert werden", str(fehler), parent=self.fenster)
                    return
                self.status.set(f"SVG exportiert · {Path(datei).name}")

    fenster = tk.Tk()
    Editor(fenster)
    fenster.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Wände, Türen und Bemaßung neu zeichnen und Maschinen lokal platzieren.")
    parser.add_argument("datei", nargs="?", help="PNG, JPG, WebP, PDF oder gespeichertes Projekt")
    parser.add_argument("--seite", type=int, help="PDF-Seitennummer, beginnend bei 1")
    parser.add_argument("--svg", metavar="DATEI", help="SVG direkt exportieren und das Editorfenster überspringen")
    args = parser.parse_args()
    if args.seite is not None and args.seite < 1:
        parser.error("--seite muss mindestens 1 sein.")
    if args.svg:
        if not args.datei:
            parser.error("Für einen SVG-Export ist eine Eingabedatei erforderlich.")
        if Path(args.svg).resolve() == Path(args.datei).resolve():
            parser.error("SVG-Ausgabe und Eingabedatei müssen unterschiedliche Pfade haben.")
        try:
            grundriss_importieren(args.datei, seite=args.seite or 1).svg_exportieren(args.svg)
        except (OSError, ValueError, Image.DecompressionBombError) as fehler:
            parser.exit(1, f"Fehler: {fehler}\n")
    else:
        _editor_starten(args.datei, args.seite)


if __name__ == "__main__":
    main()
