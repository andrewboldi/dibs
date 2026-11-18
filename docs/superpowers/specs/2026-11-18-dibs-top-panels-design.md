# dibs top as an instrument panel (#122)

The maintainer asked for a significantly more aesthetic `dibs top` and picked, from three mockups
(a timeline-first booking board, an instrument panel, a quiet typographic layout), the instrument
panel: rounded panels with their titles in the borders, like btop.

## What it shows

- **dibs**: a meter each for CPU, RAM and each GPU. Jobs' reservations are drawn in their colours,
  what programs outside dibs use in grey (#120), then what's free. The figures sit beside each
  meter. A load row is a sparkline of the machine's busy cores over the last 10 minutes, the newest
  on the right. The clock is in the top border; the GPU name and the guard mode are in the bottom
  border.
- **running N**: one row per job: its id, name, session (on terminals of 114 columns or more), CPUs,
  RAM and VRAM under labels in the panel's top border, when it ends, and a bar of its time so far
  against its estimate. A job past its estimate gets a full bar and "over by X", in gold.
- **waiting N**: the same columns and when it starts, then its slot on the plan's timeline, whose
  time axis is the panel's top border, then why it waits. The reason leaves out the RAM in use
  outside dibs, which the mem meter already shows.
- **guard**: what the guard stopped or left alone, only when there is something.
- Panels that can't show every job say "N more" in their bottom border.

## Kept from #118 and #120

- Fixed column widths with a gap that's always there.
- Meter widths that depend only on the terminal's width and the totals, so nothing jitters.
- An ellipsis wherever text is cut, and a reason only when at least 10 characters of it fit.
- The fixed 256-colour job hues.
- Each row's colour on a mark, with its text in plain ink.

New:

- Borders are a fixed mid grey (`38;5;244`), at least 3.5:1 on light and dark backgrounds. Thin
  lines need more than "dim" to show.
- Every boxed line is fitted to the inside width and padded before its borders are added, so a
  border can't drift.

## The load row

The daemon samples the machine's busy core-seconds every 4 s and keeps 10 minutes. An interval the
machine slept through is a gap. The samples are separate from the outside-CPU sampler, which resets
when jobs start or end. Both run after scheduling, and a failure in either is logged once and never
stops a tick. History starts empty when the daemon restarts; the row appears from two samples.
