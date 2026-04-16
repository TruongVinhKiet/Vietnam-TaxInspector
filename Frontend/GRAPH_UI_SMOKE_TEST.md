# Graph UI Smoke Test Checklist

## Scope
- Page: `Frontend/pages/graph.html`
- Script: `Frontend/js/graph.js`
- Goal: quick regression check after Graph UI changes.

## Prerequisites
1. Backend is running on `http://localhost:8000`.
2. User is logged in.
3. Open Graph page in browser.

## A. Workbench Navigation
1. Click tab `Doanh nghiệp`.
Expected: company table is visible, investigation section hidden.
2. Click tab `Bản đồ`.
Expected: graph investigation section appears and scrolls into view.
3. Click tab `Forensic`.
Expected: forensic panel mode is visible (focused view), company section hidden.
4. Keyboard test on workbench tabs:
- Focus a workbench tab.
- Press `ArrowLeft` or `ArrowRight`, `Home`, `End`.
Expected: active tab updates and corresponding section changes.

## B. Global Shortcuts
1. Press `G`.
Expected: switch to Graph mode.
2. Press `F`.
Expected: switch to Forensic mode.
3. Press `C`.
Expected: switch to Companies mode.
4. Press `Esc` while graph is highlighted.
Expected: highlight/fade state is cleared.

## C. Search and Graph Render
1. Type at least 2 chars in MST search.
Expected: dropdown suggestions appear.
1.1 Click outside the suggestion dropdown.
Expected: dropdown closes immediately.
2. Click a suggestion or press Enter on search input.
Expected: graph loads, loader shows then disappears.
3. Check model intelligence strip.
Expected: values for mode, thresholds, attention, depth are updated.
4. Check quality pills.
Expected: overall/serving/stress/drift pills show text state (not color-only).
5. Check investigation summary cards.
Expected: cards show company count, shell count, cycle count, and risk level.
6. Check summary hint text.
Expected: displays suspicious amount or top-attention edge context.
7. Search with a tax code that has no graph data.
Expected: empty state appears with contextual message instead of blank canvas.

## D. Graph Canvas Controls
1. Click `fit` button.
Expected: graph is centered/fitted in viewport.
2. Click `reset` button.
Expected: zoom returns to default.
3. Click a node to highlight neighborhood, then click `clear focus`.
Expected: all nodes/edges return to normal opacity.

## E. Timeline Filter
1. Move timeline range slider from `T12` to `T3`.
Expected: displayed edge intensity is filtered by month threshold, live edge count updates.
2. Click timeline play button.
Expected: month label auto-advances and play icon changes to pause.
3. Change slider while autoplay is running.
Expected: autoplay stops and slider position takes effect immediately.
4. Switch mode away from Graph while autoplay is running.
Expected: autoplay stops automatically.

## F. Forensic Panel
1. In panel tabs, switch between `NHẬT KÝ TRUY VẾT` and `CHUỖI BẰNG CHỨNG`.
Expected: smooth transition, correct pane visibility.
2. Keyboard on forensic tabs:
- Focus forensic tab.
- Press `ArrowLeft` or `ArrowRight`, `Home`, `End`.
Expected: selected forensic tab changes correctly.
3. If evidence paths exist, click `Focus Chain`.
Expected: graph focuses related nodes/edges.

## G. Action Feedback Toasts
1. Click `Xuất Báo Cáo` or `Đánh dấu` in page header.
Expected: toast appears at bottom-right with contextual message.
2. Click `YÊU CẦU GIẢI TRÌNH` or `NIÊM PHONG HỒ SƠ`.
Expected: toast appears with correct tone and auto-dismisses.

## H. Accessibility Baseline
1. Navigate with keyboard (Tab/Shift+Tab).
Expected: visible focus ring appears on interactive controls.
2. Screen reader semantic spot-check:
- Workbench has tablist semantics.
- Forensic tabs expose selected state.
- Timeline play button exposes pressed state.
3. Reduced motion preference (`prefers-reduced-motion`) check.
Expected: major animations are reduced/disabled.

## I. Company Table
1. Type in table search box.
Expected: filtering updates rows and pagination info.
2. Click `Phân tích` from a row.
Expected: MST is injected to search and graph mode opens.

## Pass Criteria
- No console errors in browser devtools.
- All sections above behave as expected.
- No blocked interaction in keyboard-only navigation.
