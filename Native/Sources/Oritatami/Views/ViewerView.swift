import SwiftUI
import WebKit

/// SwiftUI wrapper around WKWebView that hosts the standalone Mol* viewer page.
struct ViewerView: NSViewRepresentable {
    @Binding var cifURL: URL?
    var style: String     = "cartoon"
    var colorMode: String = "chain"
    var isSpin: Bool      = false
    var onHover: ((String?) -> Void)?
    var onSelect: ((ResidueInfo?) -> Void)?

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    func makeNSView(context: Context) -> WKWebView {
        let cfg = WKWebViewConfiguration()
        cfg.userContentController.add(context.coordinator, name: "viewer")
        cfg.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")

        let wv = WKWebView(frame: .zero, configuration: cfg)
        wv.setValue(false, forKey: "drawsBackground")
        context.coordinator.webView = wv
        wv.load(URLRequest(url: BackendManager.baseURL.appendingPathComponent("/viewer.html")))
        return wv
    }

    func updateNSView(_ wv: WKWebView, context: Context) {
        let coord = context.coordinator
        coord.parent = self     // keep parent reference current

        // Structure load/clear
        if cifURL != coord.lastCif {
            coord.lastCif = cifURL
            if let url = cifURL {
                coord.loadStructure(url: url)
            } else {
                coord.clearStructure()
            }
        }

        // Viewer style commands — send only on change
        if style != coord.lastStyle {
            coord.lastStyle = style
            coord.send(cmd: "style", value: "'\(style)'")
        }
        if colorMode != coord.lastColor {
            coord.lastColor = colorMode
            coord.send(cmd: "color", value: "'\(colorMode)'")
        }
        if isSpin != coord.lastSpin {
            coord.lastSpin = isSpin
            coord.send(cmd: "spin", value: isSpin ? "true" : "false")
        }
    }

    // MARK: - Coordinator

    final class Coordinator: NSObject, WKScriptMessageHandler {
        var parent: ViewerView
        weak var webView: WKWebView?
        var isReady = false
        var lastCif: URL?
        var lastStyle = "cartoon"
        var lastColor = "chain"
        var lastSpin  = false
        private var pendingLoad: URL?

        init(parent: ViewerView) { self.parent = parent }

        func userContentController(_ ctrl: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            guard message.name == "viewer",
                  let body = message.body as? [String: Any],
                  let event = body["event"] as? String else { return }

            DispatchQueue.main.async {
                switch event {
                case "ready":
                    self.isReady = true
                    if let p = self.pendingLoad { self.loadStructure(url: p); self.pendingLoad = nil }
                case "hover":
                    self.parent.onHover?(body["label"] as? String)
                case "click":
                    if let chain = body["chain"] as? String,
                       let seq   = body["seqId"]  as? Int {
                        let res = ResidueInfo(chain: chain, seqId: seq,
                                              compId: body["compId"] as? String)
                        self.parent.onSelect?(res)
                    }
                default: break
                }
            }
        }

        func loadStructure(url: URL) {
            guard isReady else { pendingLoad = url; return }
            let js = "window.postMessage({ cmd: 'load', items: [{ url: '\(url.absoluteString)', label: 'structure', format: 'mmcif' }] }, '*');"
            webView?.evaluateJavaScript(js)
        }

        func clearStructure() {
            webView?.evaluateJavaScript("window.postMessage({ cmd: 'clear' }, '*');")
        }

        func send(cmd: String, value: String) {
            guard isReady else { return }
            webView?.evaluateJavaScript("window.postMessage({ cmd: '\(cmd)', value: \(value) }, '*');")
        }
    }
}

struct ResidueInfo {
    var chain: String
    var seqId: Int
    var compId: String?
}
