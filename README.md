# Grundriss-Compiler

Ein lokaler Python-Editor für die Maschinen- und Möbelplanung. Er erstellt aus einer Bilddatei oder PDF-Seite eine bereinigte Zeichnung mit Wänden, Türen, Fensterlinien und erkannten Maßangaben. Erkannte Möbel und Maschinen erscheinen als freigestellte Grafiken in der Ablage und lassen sich an anderer Stelle platzieren.

## Schnellstart unter Windows

1. Auf GitHub **Code → Download ZIP** wählen und das Archiv entpacken.
2. Einen PowerShell-Tab im entpackten Ordner öffnen. Benötigt wird Python 3.10 oder neuer mit Tkinter. Getestet wurde der Editor unter Windows mit Python 3.10.
3. Die Pakete installieren und den Editor starten.

```powershell
python -m pip install pillow pypdfium2 rapidocr-onnxruntime
python .\grundriss.py
```

Die Installation lädt Python-Pakete. Die Grundrisse werden anschließend lokal verarbeitet.

## Grundriss bearbeiten

1. **Grundriss öffnen …** und eine PNG-, JPG-, WebP- oder PDF-Datei auswählen. Bei einem mehrseitigen PDF lässt sich die gewünschte Seite auswählen.
2. Mit **Maßstab setzen** zwei Punkte einer bekannten Strecke anklicken und die Länge in Metern eingeben. Damit lassen sich Maschinenmaße in Metern angeben.
3. Möbel oder Maschinen aus **Ablage und Vorschläge** in den Plan ziehen. Über **Ziehen und Größe ändern** lassen sich platzierte Objekte verschieben und an Kanten oder Ecken skalieren. Drehung, Name und Maße sind auch im Dialog **Maße und Name ändern …** verfügbar.
4. Fehlende Möbel oder Maschinen über **Objekt aus Vorlage wählen …** markieren. Fehlende Linien einer Tür oder Wand lassen sich über **Bauteil aus Vorlage übernehmen …** ergänzen. Dazu die ursprüngliche Bild- oder PDF-Datei im Editor öffnen und einen engen Rahmen um die gewünschte Stelle ziehen.
5. Mit **Projekt speichern …** den Plan samt Ablage und platzierten Objekten als bearbeitbare JSON-Datei speichern. Mit **SVG exportieren …** eine Zeichnung des aktuellen Layouts ausgeben.

Die Qualität der automatischen Erkennung hängt von Auflösung, Kontrast und Darstellungsstil ab. In kleinen oder stark komprimierten Vorlagen können Linien und Maßangaben fehlen. Eng benachbarte Objekte können in einem Vorschlag zusammengefasst werden.

## SVG-Export per Kommandozeile

```powershell
python .\grundriss.py "Hallenplan.pdf" --seite 2 --svg "Hallenplan.svg"
```

Als Eingabe ist auch eine gespeicherte JSON-Projektdatei möglich. Dann enthält der Export die darin platzierten Objekte einschließlich ihrer Grafiken.
