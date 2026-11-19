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
- **A panel per agent** (#167). The title is the agent's own session name where its CLI records
  one (Claude Code keeps it in ~/.claude/sessions, keyed by the session id dibs stores with each
  job), else dibs's label (project·session), then "N running · M waiting".
  - First the agent's running jobs, one row each: id, name (as wide as the longest shown, 13 to 24
    characters as the width allows, #143), CPUs, RAM and VRAM, when it ends, and a bar of its time
    so far against its estimate. A job past its estimate gets a full bar and "over by X", in gold.
  - Then a time axis row ("now", "+5m", ...) and the agent's waiting jobs: the same columns, when
    each starts, a Gantt bar for its slot on the plan, and why it waits, in a few words: "needs RAM,
    VRAM" (its sizes are in its row, and the meters say what's in use) or "behind #918", the job
    ahead in line it would delay (#143). dibs queue keeps the planner's full sentence.
  - Every panel shares the columns and the time scale, so bars in different panels compare.
  - Panels go in order of each agent's oldest job, so they keep their places as jobs start and end.
- **A header row** under the meters names the columns once (cpus, ram, vram), as htop's does, since
  an agent's name can fill its panel's title.
- **guard**: what the guard stopped or left alone, only when there is something.
- **Scrolling** (#167): the meters and the header row stay put, and everything below them scrolls
  when it doesn't fit. A bottom line counts the jobs above and below and names the keys.

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

## Grouped by agent, and scrolling (#167)

The maintainer wanted to see jobs grouped by the agent that submitted them, and to scroll through
them all, as in htop, instead of lists cut to "N more".

- **Layout.** Shown two mockups, they picked one panel per agent over agent headers inside the
  running and waiting panels. They then said they loved the waiting panel's Gantt chart, which that
  mockup had left out. So it stays: each agent's waiting rows get a time-axis row above them, on one
  scale for every panel.
- **Names.** Session names are whatever the user typed. They are drawn without escape codes or
  other control characters. Wide characters (CJK, most emoji) are shown as "?", since each takes two
  cells and would push a border out. A name is cut to leave room for its counts, so a title never
  disappears.
- **Scrolling.** The maintainer picked a pager over an htop-style highlighted row.
  - Keys: the arrows or j/k move a line, PgUp/PgDn or space a page, Home/End or g/G to either end,
    and the mouse wheel three lines. q or Ctrl-C quits.
  - The wheel needs mouse reporting, so tmux forwards it. Like htop, that takes plain click-and-drag
    text selection, and Shift+drag still selects.
  - Key reads keep an escape sequence cut off between two reads, and the terminal's modes are put
    back on every exit.
- **Output that isn't a terminal** (a pipe or a file) gets every panel at once.
