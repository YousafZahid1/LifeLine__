import SwiftUI

struct emergency: View {
    @State private var animateCamera = false
    @State private var animateCall = false
    @State private var showScanResult = false
    @State private var detectedThreat: String = ""

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

                    Text("Emergency")
                        .foregroundColor(.red)
                        .font(.largeTitle)
                        .fontDesign(.rounded)
                        .bold()

                    Spacer()

                    Button(action: {
                        Task {
                            await scanRoom()
                        }
                    }) {
                        ZStack {
                            Circle()
                                .fill(Color.blue)
                                .frame(width: 100, height: 100)
                                .scaleEffect(animateCamera ? 1.1 : 1.0)
                                .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: animateCamera)

                            Image(systemName: "camera.viewfinder")
                                .foregroundColor(.white)
                                .font(.system(size: 32, weight: .bold))
                        }
                    }
                    .onAppear {
                        animateCamera = true
                    }
                    .alert("Scan Result", isPresented: $showScanResult, actions: {
                        Button("OK", role: .cancel) {}
                    }, message: {
                        Text(detectedThreat)
                    })

                    Spacer()

                    VStack(spacing: 12) {
                        Button(action: {
                            print("Smart Location")
                        }) {
                            ZStack {
                                Circle()
                                    .fill(Color.red)
                                    .frame(width: 120, height: 120)
                                    .scaleEffect(animateCall ? 1.1 : 1.0)
                                    .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: animateCall)

                                Image(systemName: "message.fill")
                                    .foregroundColor(.white)
                                    .font(.system(size: 40, weight: .bold))
                            }
                        }
                        .onAppear {
                            animateCall = true
                        }

                        Text("Message Emergencies")
                            .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                            .font(.headline)
                            .bold()
                    }
                    .padding(.bottom, 40)
                }
                .padding()
            }
        }
    }

    func scanRoom() async {
        guard let url = URL(string: "https://3cec5fe3df34.ngrok-free.app/lifeline")


        else {
            detectedThreat = "Invalid backend URL"
            showScanResult = true
            return
        }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)

            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let responseText = json["response"] as? String {
                DispatchQueue.main.async {
                    detectedThreat = responseText
                    showScanResult = true
                }
            } else {
                DispatchQueue.main.async {
                    detectedThreat = "Unexpected response from server."
                    showScanResult = true
                }
            }
        } catch {
            DispatchQueue.main.async {
                detectedThreat = "Failed to connect to backend: \(error.localizedDescription)"
                showScanResult = true
            }
        }
    }
}

#Preview {
    emergency()
}
