
import SwiftUI

struct ContentView: View {
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

                VStack(spacing: 30) {
                    // Header section
                    VStack(spacing: 8) {
                        Spacer()
                        Spacer()
                        Spacer()
                        
                        Text("LifeLine")
                            .font(.system(size: 36, weight: .bold))
                            .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                        
                        Text("Your Personal Safety Companion")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                            .multilineTextAlignment(.center)
                        
                        Spacer()
                    }
                    .frame(height: 100)
                    Spacer()
                    
                    // Card container
                    VStack(spacing: 25) {
                        // Emergency Button Card
                        NavigationLink(destination: emergency()) {
                            HStack {
                                ZStack {
                                    Circle()
                                        .fill(Color.red.opacity(0.9))
                                        .frame(width: 50, height: 50)
                                    
                                    Image(systemName: "exclamationmark.triangle.fill")
                                        .foregroundColor(.white)
                                        .font(.system(size: 22, weight: .bold))
                                }
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Emergency Button")
                                        .font(.system(size: 18, weight: .semibold))
                                        .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                                    
                                    Text("Quick access to emergency services")
                                        .font(.system(size: 14))
                                        .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                }
                                
                                Spacer()
                                
                                Image(systemName: "chevron.right")
                                    .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                    .font(.system(size: 14, weight: .medium))
                            }
                            .padding(20)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color.white.opacity(0.8))
                                    .shadow(color: Color.blue.opacity(0.1), radius: 8, x: 0, y: 2)
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        // Smart Assistant Card
                        NavigationLink(destination: ai_call()) {
                            HStack {
                                ZStack {
                                    Circle()
                                        .fill(Color(red: 0.3, green: 0.5, blue: 0.8))
                                        .frame(width: 50, height: 50)
                                    
                                    Image(systemName: "brain.head.profile")
                                        .foregroundColor(.white)
                                        .font(.system(size: 22, weight: .bold))
                                }
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Smart Assistant")
                                        .font(.system(size: 18, weight: .semibold))
                                        .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                                    
                                    Text("AI-powered help and guidance")
                                        .font(.system(size: 14))
                                        .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                }
                                
                                Spacer()
                                
                                Image(systemName: "chevron.right")
                                    .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                    .font(.system(size: 14, weight: .medium))
                            }
                            .padding(20)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color.white.opacity(0.8))
                                    .shadow(color: Color.blue.opacity(0.1), radius: 8, x: 0, y: 2)
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        // Settings Card
                        NavigationLink(destination: settings()) {
                            HStack {
                                ZStack {
                                    Circle()
                                        .fill(Color(red: 0.5, green: 0.6, blue: 0.7))
                                        .frame(width: 50, height: 50)
                                    
                                    Image(systemName: "cross.fill")
                                        .foregroundColor(.white)
                                        .font(.system(size: 22, weight: .bold))
                                }
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Emergency Contact")
                                        .font(.system(size: 18, weight: .semibold))
                                        .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                                    
                                    Text("Add Emergencies")
                                        .font(.system(size: 14))
                                        .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                }
                                
                                Spacer()
                                
                                Image(systemName: "chevron.right")
                                    .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                    .font(.system(size: 14, weight: .medium))
                            }
                            .padding(20)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color.white.opacity(0.8))
                                    .shadow(color: Color.blue.opacity(0.1), radius: 8, x: 0, y: 2)
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        // Social Justice Resources Card
                        NavigationLink(destination: socialJusticeResources()) {
                            HStack {
                                ZStack {
                                    Circle()
                                        .fill(Color(red: 0.8, green: 0.4, blue: 0.5))
                                        .frame(width: 50, height: 50)
                                    
                                    Image(systemName: "hand.raised.fill")
                                        .foregroundColor(.white)
                                        .font(.system(size: 22, weight: .bold))
                                }
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Social Justice Resources")
                                        .font(.system(size: 18, weight: .semibold))
                                        .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                                    
                                    Text("Community safety map and resources")
                                        .font(.system(size: 14))
                                        .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                }
                                
                                Spacer()
                                
                                Image(systemName: "chevron.right")
                                    .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                    .font(.system(size: 14, weight: .medium))
                            }
                            .padding(20)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color.white.opacity(0.8))
                                    .shadow(color: Color.blue.opacity(0.1), radius: 8, x: 0, y: 2)
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                    }
                    .padding(.horizontal, 20)
                    
                    Spacer()
                    
                    // Footer
                    Text("© 2025 Created by Yousaf Zahid \n \t \tHack the Nest")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                        .padding(.bottom, 40)
                }
            }
        }
    }
}

// Preview
#Preview {
    ContentView()
}

// MARK: - App Entry Point
@main
struct self_app_congressionalApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
