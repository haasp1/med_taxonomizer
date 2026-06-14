# Local taxonomy reviewer

The reviewer is optional paper-support tooling for checking a taxonomy tree before full-corpus labeling. It is designed to run locally on the user's machine. The repository does not include project datasets, study records, or reviewer decisions.

## What it does

- Loads a taxonomy JSON file in the browser.
- Displays node labels, paths, definitions, parent information, and support counts when present.
- Lets a reviewer mark each node as `approve`, `rename`, `merge`, `split`, or `reject`.
- Exports a `review_queue.csv` file with decisions and notes.
- Runs entirely in the browser session. No server upload is required.

## Start the browser reviewer

Zero-build option:

```bash
cd apps/reviewer
python3 -m http.server 5173
```

Then open:

```text
http://127.0.0.1:5173/standalone.html
```

React/Vite development option:

```bash
cd apps/reviewer
npm install
npm run dev
```

Open the printed local URL, usually:

```text
http://127.0.0.1:5173/
```

Build a static version:

```bash
npm run build
npm run preview
```

## Create a review packet from the command line

If you prefer spreadsheet review, generate a review folder from a taxonomy JSON file:

```bash
./scripts/review_taxonomy.sh \
  --taxonomy outputs/<run_id>/taxonomy_tree_final.json \
  --output-dir outputs/<run_id>/review
```

The command writes:

- `review_queue.csv`
- `taxonomy_tree_curated_template.json`
- `review_instructions.md`

## Input format

The reviewer accepts common taxonomy JSON shapes. It looks for node-like objects with fields such as:

- `id`, `node_id`, `stable_id`, or `code`
- `label`, `name`, `title`, or `category`
- `definition` or `description`
- `path`, `taxonomy_path`, or `full_path`
- nested `nodes`, `children`, `categories`, `subcategories`, or `leaves`

This keeps the reviewer usable with taxonomy exports from this toolkit as well as with manually prepared taxonomy JSON files.

## Privacy note

Use de-identified taxonomy files. The app is local and does not upload files, but review notes and exported CSV files are still user-controlled project artifacts and should be handled according to the user's local data-governance requirements.
