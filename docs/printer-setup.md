# Printer Setup Guide

Dont-Blink works best when your video contains a repeatable **"clean moment"** — a point in each layer where the printhead is parked, retracted, or out of the camera's view. This guide shows how to create that moment on any printer.

---

## Option 1: Built-in timelapse mode (easiest)

Many modern printers and slicers have a built-in timelapse mode that parks the printhead automatically. If yours does, just enable it — Dont-Blink will find the parked frames.

| Printer / Slicer | Setting | Notes |
|---|---|---|
| **Bambu Lab** (Bambu Studio) | Print Settings → Timelapse → "Smooth" | Parks the head and takes a snapshot. Dont-Blink can use the raw video too. |
| **Creality K1 series** | Built-in timelapse in Creality Print | Similar park behavior. |
| **OctoPrint + Octolapse** | Octolapse plugin settings | Octolapse already parks; Dont-Blink is an alternative if you don't want the plugin overhead. |

If your printer already parks the head for timelapse, you're done. Just record and process.

---

## Option 2: Add a G-code park snippet

If your printer doesn't park automatically, you can add a small G-code snippet that moves the printhead to a parking position at each layer change. This adds minimal print time (~1-2 seconds per layer) and gives Dont-Blink clean frames to work with.

### Cura

Go to **Extensions → Post Processing → Modify G-Code → Add a script → Insert at layer change**

Or: **Settings → Printer → Machine Settings → After Layer Change G-code**

Add:

```gcode
; --- Dont-Blink timelapse park ---
G91                  ; Relative positioning
G1 Z0.5 F600        ; Lift nozzle slightly to avoid stringing
G90                  ; Absolute positioning
G1 X0 Y220 F9000    ; Move to park position (front-left corner)
G4 P300             ; Wait 400ms for camera to capture
G1 Z{layer_height}  ; Return to print height
; --- End park ---
```

**Adjust `X0 Y220`** to a corner or edge position that works for your printer's bed size. The goal is to move the head to the same spot every time.

### PrusaSlicer / OrcaSlicer

Go to **Printer Settings → Custom G-code → After layer change G-code**

Add:

```gcode
; --- Dont-Blink timelapse park ---
G91                  ; Relative positioning
G1 Z0.5 F600        ; Lift nozzle
G90                  ; Absolute positioning
G1 X0 Y{print_bed_max[1]} F9000  ; Park at front-left
G4 P400             ; Wait for camera
; --- End park ---
```

`{print_bed_max[1]}` automatically uses your printer's max Y value.

### Klipper (macro)

Add to your `printer.cfg`:

```ini
[gcode_macro TIMELAPSE_PARK]
gcode:
    {% set PARK_X = 0 %}
    {% set PARK_Y = printer.toolhead.axis_maximum.y - 5 %}
    SAVE_GCODE_STATE NAME=TIMELAPSE_PARK
    G91
    G1 Z0.5 F600
    G90
    G1 X{PARK_X} Y{PARK_Y} F9000
    G4 P400
    RESTORE_GCODE_STATE NAME=TIMELAPSE_PARK
```

Then in your slicer's "After layer change G-code," add:

```gcode
TIMELAPSE_PARK
```

### Marlin (direct G-code)

If you don't use a slicer that supports custom layer-change G-code, you can use a post-processing script or add this to your start G-code as a macro. The G-code itself is the same as the Cura example above.

---

## Option 3: Camera framing (no G-code needed)

If you don't want to modify G-code at all, you can position your camera so the printhead naturally moves **out of frame** during travel moves:

- **Mount the camera low and close**, angled upward at the print. The printhead moves out of the top of the frame during long travel moves.
- **Use a telephoto/zoom lens** (or crop) to frame just the print bed area. Travel moves take the head outside the crop.
- **Position the camera on the opposite side** from where the printer homes or does its longest travel.

This approach depends on your printer's travel patterns and won't work for every setup, but when it does, it requires zero configuration.

---

## Tuning tips

### Every layer vs. every N layers

Parking every layer gives the smoothest timelapse but adds ~1-2 seconds per layer. For a 500-layer print, that's ~8-16 minutes of extra print time.

To park less often (e.g., every 5 layers), wrap the park G-code in a conditional:

**Klipper:**

```ini
[gcode_macro TIMELAPSE_PARK]
gcode:
    {% if printer["gcode_move"].position.z|int % 1 == 0 %}  ; Adjust modulo for frequency
    ; ... park code ...
    {% endif %}
```

**PrusaSlicer/OrcaSlicer** (using layer number):

Most slicers expose `{layer_num}` or `{layer_z}` — check your slicer's documentation for conditional G-code support.

### Park position choice

- **Corner parking** (e.g., X0 Y_max): Most reliable. The head goes to the same spot every time.
- **Side parking** (e.g., X0 Y_current): Faster travel but the head is still partially visible on some camera angles.
- **Rear parking**: Good if your camera faces the front of the printer.

The best park position depends on where your camera is. Experiment with a short test print.

When processing, tell Dont-Blink where you park: `--capture-mode left-park`, `--capture-mode right-park`, or `--capture-mode top-park` (for overhead cameras). This gives cleaner results than auto-detection.

### Dwell time (G4)

The `G4 P400` (400ms pause) gives the camera time to capture a clean frame. If your camera records at 30 FPS, even `G4 P200` (200ms) gives ~6 frames of clean footage. Increase if your camera has slow auto-exposure.

---

## Still not working?

If Dont-Blink produces very few frames or inconsistent results:

1. **Tell the tool where the head parks:** If your G-code parks the head on the left, use `--capture-mode left-park`. For right or top, use `right-park` or `top-park`. If the head moves out of frame, use `out-of-view`. This is often the most reliable fix.
2. Run `dontblink visualize-video input.mp4 debug.mp4` to see what the model detects in each frame.
3. Check that the park position is actually consistent — watch the debug video.
4. Try lowering the confidence threshold in your config: `detection.confidence: 0.3`.
5. Open an issue with `dontblink doctor --copy` output and a few sample frames.
