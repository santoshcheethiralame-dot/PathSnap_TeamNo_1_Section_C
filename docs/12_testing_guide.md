# 12 · Testing guide

Everything here was run on one machine on 18 Nov 2026; the real two-laptop run (part D) has not been done yet. Commands are for Windows (cmd or PowerShell) from the repo root, with the venv active.

## A. Code and unit tests (any machine, no data needed)
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests
```
Expect `20 passed` (about 5 s). They cover the loader, segments, label alignment, candidates, features, the baseline, the HMM, route overlap, and the API on a synthetic trip (list, predict, scoring, unscored trip, bad ids, empty folder).

## B. Numbers reproduce (needs the organiser data in `data/raw/`)
```bash
python scripts/check_predict.py data/raw
python scripts/route_overlap.py data/raw DATA_PROCESSED PROBS_DIR out.csv
```
`check_predict.py` runs `predict_trip` exactly as the API does. Expect point-weighted Top-1 **0.9315**, mean route continuity **0.904**, at most about 2.4 s per trip. If these differ, something in the pipeline or the saved model changed.

## C. The API on one machine
Start it (set the trips folder first if it is not `../Student folder/student_test`):
```bash
set PATHSNAP_TRIPS_DIR=C:\path\to\trips
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
The console prints `Serving 10 trips from ...`. In another terminal:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/trips
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\":\"00000040\"}"
```
(In PowerShell use `Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType application/json -Body '{"trip_id":"00000040"}'`.)

Expected `/predict` results for the ten student-test trips (tuned boosting + HMM; use these to check Sharath's setup):

| Trip | Top-1 | Route continuity |
|---|---|---|
| 00000000 | 0.963 | 0.939 |
| 00000007 | 0.993 | 0.992 |
| 00000009 | 0.887 | 0.855 |
| 00000040 | 0.955 | 0.957 |
| 00000052 | 0.992 | 0.877 |
| 00000056 | 0.945 | 0.884 |
| 00000065 | 0.902 | 0.909 |
| 00000076 | 0.956 | 0.885 |
| 00000077 | 0.864 | 0.894 |
| 00000091 | 0.928 | 0.849 |

Prediction coverage is 1.0 on all ten. The first request after start is slower (model and graph load). **Warm up by calling each trip once before the demo.** Every request appears in the console and in `api/logs/requests.log` with trip id, method, route size, Top-1, continuity and time.

Bad input: `{"trip_id":"../x"}` gives 400, an unknown id 404, a missing id 422.

## D. The two-laptop test (the one still to do on real hardware)
Laptop 1 runs the API, Laptop 2 the UI.
1. **Network.** Put both on the same Wi-Fi or one phone hotspot. Some campus and guest networks isolate clients; if pings between the laptops fail, use a hotspot.
2. **Laptop 1 address.** `ipconfig`, take the IPv4 address of the Wi-Fi adapter (for example 192.168.1.20), not a VPN or virtual-machine adapter.
3. **Firewall.** Check the network profile with `Get-NetConnectionProfile` (PowerShell). This laptop's profile is **Public**, so a rule limited to Private would not apply. Either switch the network to Private or open the port for every profile, in an administrator terminal:
   `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000 profile=any`
   (Windows may also pop up an "allow Python" prompt the first time the API starts: allow it.)
4. **Start the API** on Laptop 1 with `--host 0.0.0.0` (not the default 127.0.0.1, which only accepts local calls).
5. **From Laptop 2, check the API before the UI:** open `http://192.168.1.20:8000/health` in a browser. You should see `{"status":"ok","trips_available":10}`. If this fails, the problem is network or firewall, not our code.
6. **UI on Laptop 2.** Copy the `ui/` folder and run `python -m http.server 8080 --directory ui`, open `http://localhost:8080`, type `http://192.168.1.20:8000` in the address box, press **Load Trips**, choose a trip, press **Reconstruct Route**. Expect: status "10 trips loaded successfully.", a blue predicted route over the orange fixes, and the four cards (Top-1, coverage, continuity, arcs) matching table C.
7. **Watch Laptop 1's console** while you click. Each click should add a `POST /predict` line and a `predict trip=...` line. This is the log the evaluators will ask to see.
8. **Take the screenshots for report §7:** the UI with a result, and Laptop 1's console with successful requests.

Troubleshooting:

| Symptom | Likely cause |
|---|---|
| UI says "Could not connect to the API" | wrong address or port, API not running, firewall, or Laptop 2 on a different network |
| `/health` times out from Laptop 2 | Public network profile or missing firewall rule |
| `/health` works locally but not remotely | API started without `--host 0.0.0.0` |
| Browser console shows a CORS error | should not happen (CORS is open); check you are calling the API port, not the UI port |
| `/trips` returns `count: 0` | `PATHSNAP_TRIPS_DIR` wrong or not set in that terminal |
| Map is blank grey | no internet on the UI laptop (Leaflet and tiles load from the web) |
| 500 on `/predict` | read the traceback in Laptop 1's console |

## E. Robustness (what the evaluator trips might throw at it)
- **Trip without `.route`:** returns the route with `"scored": false`; no error. Checked.
- **Defective training trips run through the API:** trip 33 (5,844 points) 200 in 5.2 s; trip 54 (19,313 points, fixes far from the network) 200 in 6.9 s with coverage 0.42 and continuity 0.59, so it degrades instead of crashing; trip 16 200 in 2.6 s. The accuracy numbers on training trips are in-sample (the final model saw them), so do not quote them.
- **Restart:** stop and restart the API; the next request works (no state kept).
- **Four requests at once** (trips 56, 77, 40, 0): all returned 200, each took 4.3 to 5.9 s instead of the usual 0.4 to 1.9 s, because prediction is CPU-bound and the requests share the CPU. Fine for one click at a time; do not click through trips faster than the results arrive.
- **API stopped:** the UI shows "Prediction could not be completed. Check the backend connection." and "Could not connect to the API..." and recovers without a page reload once the API is back (checked; first call after a restart 1.7 s).
- **Not tested:** a very large trip (more than 20,000 points), more than 10 trips, a trip folder with extra files.

## F. Checks before submitting
- `git status` clean, tests green on a fresh clone (`git clone`, new venv, section A).
- No dataset or secrets in the repo: `git ls-files | findstr /i ".track .nodes .arcs .route .env"` returns nothing; scan the history too (done once on 18 Nov, repeat before the final push).
- Every number in the report appears in a CSV in `results/` and its script is in the README.
- Nothing from the evaluator partition used in development: it cannot be listed by `pathsnap.trips` by design, and the API only sees it through `PATHSNAP_TRIPS_DIR`.
- Viva: each member runs section C and D once themselves and can explain `predict_trip` step by step.

## G. One-laptop rehearsal, run on 18 Nov 2026 (fresh clone of the pushed `main`)
- 20 of 20 tests pass in the clone.
- API and UI started as two separate servers (ports 8000 and 8080), API bound to all interfaces. `/health` answered on the laptop's LAN address (10.30.201.190) as well as on localhost, and a CORS preflight from the UI's origin was accepted.
- UI driven in the built-in browser (page on `localhost:8080`, API on `localhost:8000`, so two origins): Load Trips gave 10 trips; Reconstruct Route on all ten showed Top-1 and continuity cards equal to the table in section C, coverage 100%, a predicted route and the observed fixes drawn on the map, 0.2 to 1.9 s per trip.
- Bad ids 400, 404 and 422 as expected; 37 log lines, 14 `predict trip=` lines, no errors or tracebacks.
- Not testable on one laptop: the real network path, client isolation and the firewall between two machines. The built-in browser pane refuses to load pages from the LAN address (`ERR_BLOCKED_BY_CLIENT`), so the browser run used `localhost`; the LAN address was checked with curl.
