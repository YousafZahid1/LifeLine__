import SwiftUI

struct settings: View {
    @State private var phoneNumber: String = ""
    @State private var role: String = ""
    @State private var contacts: [(role: String, phone: String)] = []

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
                
                VStack(spacing: 40) {
                    Spacer()
                    
                    Text("Emergency Contact")
                        .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .padding(.top)
                    
                    Spacer()
                    
                    VStack(spacing: 20) {
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color.white.opacity(0.8))
                            .frame(height: 60)
                            .shadow(color: Color.blue.opacity(0.1), radius: 8, x: 0, y: 2)
                            .overlay(
                                TextField("Phone Number", text: $phoneNumber)
                                    .keyboardType(.phonePad)
                                    .padding(.horizontal)
                                    .foregroundColor(Color(.darkGray))
                                    .font(.system(size: 16, weight: .semibold))
                            )
                        
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color.white.opacity(0.8))
                            .frame(height: 60)
                            .shadow(color: Color.blue.opacity(0.1), radius: 8, x: 0, y: 2)
                            .overlay(
                                TextField("Role (Mom, Dad, etc.)", text: $role)
                                    .padding(.horizontal)
                                    .foregroundColor(.black)
                                    .font(.system(size: 16, weight: .bold)) // Add bold weight
                            )
                        
                        Button(action: {
                            guard !phoneNumber.isEmpty, !role.isEmpty else { return }
                            contacts.append((role: role, phone: phoneNumber))
                            saveContacts() 
                            phoneNumber = ""
                            role = ""
                        }) {
                            Text("Add Contact")
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding()
                                .frame(maxWidth: .infinity)
                                .background(LinearGradient(
                                    colors: [Color(red: 0.3, green: 0.5, blue: 0.8), Color(red: 0.4, green: 0.6, blue: 0.9)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                ))
                                .cornerRadius(20)
                                .shadow(color: Color.blue.opacity(0.3), radius: 10, x: 0, y: 5)
                        }
                        .padding(.top, 10)
                    }
                    .padding(.horizontal, 40)
                    
                    Spacer()
                    
                    if !contacts.isEmpty {
                        VStack(spacing: 15) {
                            Text("Emergency Contacts")
                                .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                .font(.headline)
                                .padding(.top)
                            
                            ScrollView {
                                VStack(spacing: 12) {
                                    ForEach(contacts, id: \.phone) { contact in
                                        HStack {
                                            VStack(alignment: .leading, spacing: 4) {
                                                Text(contact.role)
                                                    .font(.headline)
                                                    .foregroundColor(Color(red: 0.2, green: 0.3, blue: 0.5))
                                                Text(contact.phone)
                                                    .font(.subheadline)
                                                    .foregroundColor(Color(red: 0.4, green: 0.5, blue: 0.6))
                                            }
                                            Spacer()
                                        }
                                        .padding()
                                        .background(Color.white.opacity(0.6))
                                        .cornerRadius(25)
                                        .shadow(color: Color.blue.opacity(0.1), radius: 6, x: 0, y: 2)
                                        .padding(.horizontal, 30)
                                    }
                                }
                            }
                        }
                    }
                    
                    Spacer()
                }
            }
        }
        .onAppear {
            loadContacts()
        }
    }
    
    func saveContacts() {
        let contactsToSave = contacts.map { ["role": $0.role, "phone": $0.phone] }
        UserDefaults.standard.set(contactsToSave, forKey: "savedContacts")
    }
    
    func loadContacts() {
        if let saved = UserDefaults.standard.array(forKey: "savedContacts") as? [[String: String]] {
            contacts = saved.compactMap { dict in
                if let role = dict["role"], let phone = dict["phone"] {
                    return (role: role, phone: phone)
                }
                return nil
            }
        }
    }
}

#Preview {
    settings()
}
