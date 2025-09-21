import SwiftUI

struct ai_call: View {
    @State private var animateCamera = false
    @State private var animateCall = false
    @State private var showScanResult = false
    @State private var detectedThreat: String = ""
    @State private var geminiResponse: String = ""
    
    var body: some View {
        NavigationStack {
            ZStack {
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color(red: 0.95, green: 0.97, blue: 1.0),
                        Color(red: 0.92, green: 0.95, blue: 0.98)
                    ]),
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()
                
                VStack {
                    Spacer()
                    
                    Text("Smart Assistant")
                        .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                        .font(.largeTitle)
                        .fontDesign(.rounded)
                        .bold()
                    
                    Spacer()
                    Spacer()
                    
                    BreathingView()
                        .frame(width: 200, height: 200)
                        .padding(.bottom, 20)
                    
                    Spacer()
                    
                    VStack(spacing: 12) {
                        Button(action: {
                            startAssistant()
                        }) {
                            ZStack {
                                Circle()
                                    .fill(Color(red: 0.3, green: 0.5, blue: 0.8))
                                    .frame(width: 120, height: 120)
                                    .scaleEffect(animateCall ? 1.1 : 1.0)
                                    .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: animateCall)
                                
                                Image(systemName: "brain.head.profile")
                                    .foregroundColor(.white)
                                    .font(.system(size: 40, weight: .bold))
                                    .shadow(color: Color(red: 0.3, green: 0.5, blue: 0.8), radius: 10)
                            }
                        }
                        .onAppear {
                            animateCall = true
                        }
                        
                        Text("AI Assistant")
                            .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                            .font(.headline)
                            .bold()
                        
                        if !geminiResponse.isEmpty {
                            Text("Gemini: \(geminiResponse)")
                                .foregroundColor(Color(red: 0.3, green: 0.6, blue: 0.4))
                                .multilineTextAlignment(.center)
                                .padding(.top, 10)
                        }
                    }
                    .padding(.bottom, 40)
                }
                .padding()
            }
        }
    }
    
    
    func sendPostRequest(to endpoint: String, with body: [String: Any]? = nil, completion: ((String?) -> Void)? = nil) {
        guard let url = URL(string: "http://172.16.2.243:5004/\(endpoint)") else {
            print("Invalid URL")
            completion?(nil)
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        if let body = body {
            request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("❌ Error: \(error.localizedDescription)")
                completion?(nil)
                return
            }
            if let data = data, let responseStr = String(data: data, encoding: .utf8) {
                print("✅ Response: \(responseStr)")
                completion?(responseStr)
            }
        }.resume()
    }
    
    func getStatus(completion: @escaping (String?) -> Void) {
        guard let url = URL(string: "http://172.16.2.243:5004") else {
            print("Invalid URL")
            completion(nil)
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                print("❌ Error: \(error.localizedDescription)")
                completion(nil)
                return
            }
            guard let data = data else {
                print("No data received")
                completion(nil)
                return
            }
            do {
                if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                   let lastResponse = json["last_response"] as? String {
                    completion(lastResponse)
                } else {
                    completion(nil)
                }
            } catch {
                print("❌ JSON parsing error: \(error)")
                completion(nil)
            }
        }.resume()
    }
    
    
    func startAssistant() {
        sendPostRequest(to: "start") { _ in
            Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { timer in
                getStatus { latestReply in
                    if let reply = latestReply {
                        DispatchQueue.main.async {
                            geminiResponse = reply
                        }
                        if reply.contains("You're welcome! Goodbye!") {
                            timer.invalidate()
                        }
                    }
                }
            }
        }
    }
    
    func sendChatMessage(_ message: String) {
        sendPostRequest(to: "chat", with: ["message": message]) { response in
            guard let response = response else { return }
            if let data = response.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let geminiText = json["llama_response"] as? String {
                DispatchQueue.main.async {
                    geminiResponse = geminiText
                }
            }
        }
    }
    
    func stopAssistant() {
        sendPostRequest(to: "stop") { _ in
            DispatchQueue.main.async {
                geminiResponse = "Assistant stopped"
            }
        }
    }
}

struct BreathingView: View {
    @State private var scale: CGFloat = 1.0
    @State private var showInhale = true
    
    var body: some View {
        ZStack {
            Circle()
                .fill(Color(red: 0.6, green: 0.7, blue: 0.9).opacity(0.3))
                .scaleEffect(scale)
                .animation(
                    Animation.easeInOut(duration: 4).repeatForever(autoreverses: true),
                    value: scale
                )
                .onAppear { scale = 1.3 }
            
            Text(showInhale ? "Breathe In" : "Breathe Out")
                .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                .font(.headline)
                .opacity(0.7)
                .onAppear {
                    Timer.scheduledTimer(withTimeInterval: 4, repeats: true) { _ in
                        showInhale.toggle()
                    }
                }
        }
    }
}

#Preview {
    ai_call()
}
