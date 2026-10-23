# PaperTrail AI

PaperTrail AI is a local research-paper review assistant. It extracts text from
text-searchable PDFs, heuristically identifies candidate claims, retrieves passages from
the same paper, and uses a natural-language-inference (NLI) model to help classify each
claim as supported, contradicted, insufficiently evidenced, or requiring review.

These labels are model assessments of the relationship between a passage and a claim.
They do not establish scientific truth. Claim extraction is heuristic, citations are
detected by format only, and numerical differences are review prompts rather than
confirmed errors.

The Streamlit interface accepts PDF files (up to its configured 200 MB per-file limit).
`app.py` orchestrates page parsing, preprocessing, candidate-claim extraction, evidence
retrieval, BART NLI assessment, citation/numerical checks, and CSV/PDF report generation
through the modules in `src/`.

## Windows setup

For a fresh Windows setup, Python 3.11 is available. Create an isolated environment in
PowerShell and install the project dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. This workspace was validated with its
configured Python 3.14.3 interpreter and Streamlit 1.55.0. On the validated machine,
these exact commands run the project without creating another environment:

```powershell
& 'C:\Users\gvman\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m compileall -q app.py src tests
& 'C:\Users\gvman\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m pytest -q
& 'C:\Users\gvman\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m streamlit run app.py
```

The Python 3.11 virtual environment commands above describe a fresh install and were
not run here. The existing environment's installed versions of several dependencies
differ from the older exact pins in `requirements.txt`; `pip check` reported no broken
installed requirements. The regression and integration results below apply to the
configured Python 3.14.3 environment.

## Workflow

1. Upload a text-searchable research PDF in **Research Workspace**.
2. The parser extracts page text; preprocessing normalizes whitespace and segments
   sentences while retaining page numbers.
3. Heuristics surface candidate claims. Sentence-transformer embeddings retrieve
   potentially relevant passages, with a TF-IDF fallback if embeddings are unavailable.
4. The NLI model compares each evidence passage (premise) with a claim (hypothesis).
   Conflicting or uncertain predictions require manual review.
5. Inspect the evidence, page references, citation markers, numerical review prompts,
   and CSV/PDF downloads in the app.

The **Load demo paper** action uses a small synthetic PDF and labels it as synthetic. It
is not a published paper and its output is not a verified research result.

## Audit status interpretation

- `SUPPORTED` and `CONTRADICTED` describe a sufficiently confident NLI relationship
  between retrieved in-document evidence and a candidate claim; neither proves
  scientific truth or falsehood.
- `INSUFFICIENT_EVIDENCE` means no relevant evidence was found or the NLI model did not
  establish support or contradiction. Missing evidence is not proof of falsity.
- `HUMAN_REVIEW_REQUIRED` means model predictions were unavailable, uncertain, or mixed
  across retrieved passages.
- NLI scores are the model's confidence in its selected label, not the probability that
  a scientific claim is objectively true.

## Models, internet, and hardware

The first retrieval that uses semantic embeddings may download
`sentence-transformers/all-MiniLM-L6-v2`. If it cannot be loaded, the application falls
back to local TF-IDF retrieval. The first applicable NLI audit downloads
`facebook/bart-large-mnli` (about 1.63 GB for the model weights). Ensure at least 4 GB
of free disk space before the initial model load; the Transformers cache is normally
under `%USERPROFILE%\.cache\huggingface\hub`. Weights are cached after download. The
first download needs internet access. If download or model initialization fails, the app
reports the error and marks affected claims for manual review rather than producing
fabricated model labels.

Both models can run on CPU. First-time downloads and CPU inference can take time; BART
is comparatively large and works best with several gigabytes of available memory. The
actual model is loaded lazily when the first relevant claim needs NLI.
The model's own configuration defines the label mapping; for this BART checkpoint it is
`0=contradiction`, `1=neutral`, and `2=entailment`.

### Optional OCR on Windows

The app can OCR pages that have no selectable text when **Use OCR for scanned pages**
is enabled. Tesseract OCR 5.4.0.20240606 and English (`eng`) data were installed and
verified on the validation machine. Install the executable separately from the Python
package on other machines:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

The app detects the standard `C:\Program Files\Tesseract-OCR` install path even if
the installer has not updated the current process PATH. Restart Streamlit, then verify
Tesseract and English data (adjust the executable path if installed elsewhere):

```powershell
& "$env:ProgramFiles\Tesseract-OCR\tesseract.exe" --version
& "$env:ProgramFiles\Tesseract-OCR\tesseract.exe" --list-langs
```

The language list must include `eng`. If it does not, install the English traineddata
for that Tesseract installation and verify again. The Python `pytesseract` wrapper is
installed from `requirements.txt`; it does not install the OCR executable or language
data. When either is missing, the app displays a specific error and does not claim to
have extracted scanned text. OCR runtime depends on page count and resolution.

## Limitations and troubleshooting

- Only pages without selectable text are passed to OCR; searchable page text is kept as
  extracted from the PDF. The default OCR language is English (`eng`).
- OCR errors, missing language data, corrupt PDFs, and unavailable model downloads are
  reported rather than silently represented as successful analysis.
- If BART cannot be downloaded, check internet access, free disk space, and the Hugging
  Face cache path shown above. Remove an incomplete cached model snapshot only after
  confirming the app is stopped; then retry with internet access.
- Keep Streamlit at the pinned 1.55.0 version. The download buttons use its
  `on_click="ignore"` option so a click does not invalidate the in-memory report payload
  before the browser can retrieve it.
- If OCR is unavailable, install Tesseract and language data as shown above, or provide
  a text-searchable PDF.
- Claim extraction is heuristic: it filters some paper-structure and procedure-only
  sentences but can still miss findings or surface weak candidates. Review the original
  claim and retrieved passage text and pages before relying on any status.
- The available local source PDF, *Trends of Reinforcement Learning*, is an image-only
  B.Tech presentation deck, not a research paper. OCR recovered 7,070 characters across
  its 14 pages; the claim heuristic extracted no claims from the deck. Generated report
  PDFs and the demo are synthetic project artifacts.
- A genuine-paper integration check used *Attention Is All You Need* by Ashish Vaswani
  et al., obtained from [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). The exact
  PDF was parsed as 15 pages (39,593 characters), producing 12 candidate claims, 36
  claim/evidence comparisons, 47 citation markers, and no numerical warnings. The final
  statuses were 10 `INSUFFICIENT_EVIDENCE` and 2 `HUMAN_REVIEW_REQUIRED`; none were
  classified supported or contradicted. Manual inspection found relevant passages, but
  some clear paraphrases received neutral NLI predictions or conflicted with other
  retrieved passages. This is one end-to-end sanity check, not an accuracy benchmark.
- Native Playwright download events were not exposed by the browser harness (a synthetic
  data-URL probe also produced no event). As a fallback, the actual same-origin
  Streamlit report endpoints were fetched and returned HTTP 200 with the expected
  attachment filenames. The CSV parsed as 36 rows with the expected headers; the PDF
  opened as a six-page report and contained the report title and paper claim. The route
  and content are verified; native browser download-event capture is not.

## Tests

`python -m pytest -q` runs deterministic tests without model downloads. The synthetic
pipeline test uses an explicit NLI stub to validate classification rules; it does not
represent real model predictions. The NLI unit test validates label mapping with a
controlled model output. Real model inference is an integration check and requires the
BART weights; actual CPU inference on entailment, neutral, and contradiction examples
was verified. OCR unit tests use a deterministic OCR engine stub; OCR was also run with
the installed Tesseract engine on all pages of the available scanned deck. Real OCR on
a genuine research paper remains untested. CSV/PDF payload generation is covered by
tests; the generated report routes were also fetched and parsed during browser
validation as described above.
