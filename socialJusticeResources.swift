import SwiftUI
import Foundation
import MapKit
import CoreLocation

class LocationManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    private let locationManager = CLLocationManager()
    @Published var location: CLLocation?
    @Published var authorizationStatus: CLAuthorizationStatus = .notDetermined
    @Published var isLocationAvailable = false

    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = 10
    }

    func requestLocationPermission() {
        locationManager.requestWhenInUseAuthorization()
    }

    func startLocationUpdates() {
        guard authorizationStatus == .authorizedWhenInUse || authorizationStatus == .authorizedAlways else {
            return
        }
        locationManager.startUpdatingLocation()
    }

    func stopLocationUpdates() {
        locationManager.stopUpdatingLocation()
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        self.location = location
        isLocationAvailable = true
    }

    func locationManager(_ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus) {
        authorizationStatus = status
        isLocationAvailable = (status == .authorizedWhenInUse || status == .authorizedAlways)
        
        if isLocationAvailable {
            startLocationUpdates()
        } else {
            stopLocationUpdates()
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error.localizedDescription)")
        isLocationAvailable = false
    }
}

struct Resource: Identifiable, Codable, Hashable {
    let id: UUID
    let name: String
    let type: ResourceType
    let address: String
    let latitude: Double?
    let longitude: Double?
    let riskFactor: Double
    let averageRating: Double
    let totalRatings: Int
    let incidents: [Incident]
    let environmentalFactors: [EnvironmentalFactor]
    let comments: [Comment]
    let trustIndicators: TrustIndicators

    init(name: String, type: ResourceType, address: String, latitude: Double? = nil, longitude: Double? = nil, riskFactor: Double, averageRating: Double, totalRatings: Int, incidents: [Incident], environmentalFactors: [EnvironmentalFactor], comments: [Comment], trustIndicators: TrustIndicators) {
        self.id = UUID()
        self.name = name
        self.type = type
        self.address = address
        self.latitude = latitude
        self.longitude = longitude
        self.riskFactor = riskFactor
        self.averageRating = averageRating
        self.totalRatings = totalRatings
        self.incidents = incidents
        self.environmentalFactors = environmentalFactors
        self.comments = comments
        self.trustIndicators = trustIndicators
    }
}

enum ResourceType: String, CaseIterable, Codable, Hashable {
    case communityCenter = "Community Center"
    case library = "Library"
    case park = "Park"
    case healthCenter = "Health Center"
    case shelter = "Shelter"

    var icon: String {
        switch self {
        case .communityCenter: return "building.2"
        case .library: return "book"
        case .park: return "tree"
        case .healthCenter: return "cross"
        case .shelter: return "house"
        }
    }

    var mapPinColor: Color {
        switch self {
        case .communityCenter: return .blue
        case .library: return .green
        case .park: return .orange
        case .healthCenter: return .red
        case .shelter: return .purple
        }
    }
}

struct Incident: Identifiable, Codable, Hashable {
    let id: UUID
    let type: String
    let date: Date
    let severity: IncidentSeverity
    let description: String
    let verified: Bool
    let latitude: Double?
    let longitude: Double?

    init(type: String, date: Date, severity: IncidentSeverity, description: String, verified: Bool, latitude: Double? = nil, longitude: Double? = nil) {
        self.id = UUID()
        self.type = type
        self.date = date
        self.severity = severity
        self.description = description
        self.verified = verified
        self.latitude = latitude
        self.longitude = longitude
    }
}

enum IncidentSeverity: String, CaseIterable, Codable, Hashable {
    case low = "Low"
    case medium = "Medium"
    case high = "High"

    var color: Color {
        switch self {
        case .low: return .yellow
        case .medium: return .orange
        case .high: return .red
        }
    }
}

struct EnvironmentalFactor: Identifiable, Codable, Hashable {
    let id: UUID
    let factor: String
    let impact: String
    let riskLevel: Double

    init(factor: String, impact: String, riskLevel: Double) {
        self.id = UUID()
        self.factor = factor
        self.impact = impact
        self.riskLevel = riskLevel
    }
}

struct Comment: Identifiable, Codable, Hashable {
    let id: UUID
    let author: String
    let rating: Int
    let text: String
    let date: Date
    let helpful: Int

    init(author: String, rating: Int, text: String, date: Date, helpful: Int) {
        self.id = UUID()
        self.author = author
        self.rating = rating
        self.text = text
        self.date = date
        self.helpful = helpful
    }
}

struct TrustIndicators: Codable, Hashable {
    let communityVerified: Bool
    let officialPartnership: Bool
    let regularUpdates: Bool
    let transparentReporting: Bool

    var score: Double {
        let indicators = [communityVerified, officialPartnership, regularUpdates, transparentReporting]
        return Double(indicators.filter { $0 }.count) / Double(indicators.count)
    }
}

public struct socialJusticeResources: View {
    @StateObject private var locationManager = LocationManager()
    @State private var resources: [Resource] = []
    @State private var selectedResource: Resource?
    @State private var searchText = ""
    @State private var showingAddIncident = false
    @State private var showingAddComment = false
    @State private var region = MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 38.91194, longitude: -77.22244), // Vienna, VA - 1934 Old Gallows Rd
        span: MKCoordinateSpan(latitudeDelta: 0.01, longitudeDelta: 0.01) // Closer zoom for better detail
    )
    @State private var selectedResourceType: ResourceType? = nil
    @State private var isLoading = false
    @State private var errorMessage: String?

    var filteredResources: [Resource] {
        var filtered = resources

        if let selectedType = selectedResourceType {
            filtered = filtered.filter { $0.type == selectedType }
        }

        if !searchText.isEmpty {
            filtered = filtered.filter {
                $0.name.localizedCaseInsensitiveContains(searchText) ||
                $0.address.localizedCaseInsensitiveContains(searchText)
            }
        }

        return filtered.sorted { $0.riskFactor < $1.riskFactor }
    }

    public var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Search Bar
                SearchBar(text: $searchText, onSearchButtonClicked: {
                    Task {
                        await searchResources()
                    }
                })

                // Filter Buttons
                FilterBarView(selectedType: $selectedResourceType)
                
                // Location Status Indicator
                if !locationManager.isLocationAvailable && locationManager.authorizationStatus != .notDetermined {
                    HStack {
                        Image(systemName: "location.slash")
                            .foregroundColor(.orange)
                        Text("Location access needed for better experience")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Button("Enable") {
                            locationManager.requestLocationPermission()
                        }
                        .font(.caption)
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 4)
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(8)
                    .padding(.horizontal)
                }

                // Map View
                ZStack(alignment: .topTrailing) {
                    MapView(
                        resources: filteredResources,
                        selectedResource: $selectedResource,
                        region: $region
                    )
                    .frame(height: 300)
                    .cornerRadius(12)
                    
                    // Center on location button
                    if locationManager.isLocationAvailable {
                        Button(action: centerOnUserLocation) {
                            Image(systemName: "location.fill")
                                .font(.title2)
                                .foregroundColor(.white)
                                .frame(width: 44, height: 44)
                                .background(Color.blue)
                                .clipShape(Circle())
                                .shadow(radius: 4)
                        }
                        .padding(.top, 8)
                        .padding(.trailing, 8)
                    }
                }
                .padding(.horizontal)

                // Resource List
                List(filteredResources) { resource in
                    ResourceRowView(resource: resource)
                        .onTapGesture {
                            selectedResource = resource
                            if let lat = resource.latitude, let lon = resource.longitude {
                                region.center = CLLocationCoordinate2D(latitude: lat, longitude: lon)
                            }
                        }
                }
                .searchable(text: $searchText, prompt: "Search by location or name...")
            }
            .navigationTitle("Community Safety Map")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button("Add Incident") {
                            showingAddIncident = true
                        }
                        Button("Add Review") {
                            showingAddComment = true
                        }
                        Button("Refresh") {
                            Task {
                                await loadResources()
                            }
                        }
                    } label: {
                        Image(systemName: "plus.circle")
                    }
                }
            }
        }
        .task {
            locationManager.requestLocationPermission()
            await loadResources()
        }
        .sheet(isPresented: $showingAddIncident) {
            AddIncidentView(resources: resources) { incident in
                addIncident(incident)
            }
        }
        .sheet(isPresented: $showingAddComment) {
            AddCommentView(resources: resources) { comment, resource in
                addComment(comment, to: resource)
            }
        }
        .sheet(item: $selectedResource) { resource in
            ResourceDetailView(resource: resource)
        }
    }

    @MainActor
    private func loadResources() async {
        isLoading = true
        errorMessage = nil

        // Load mock data for now
        resources = MockData.sampleResources

        // Update map region to user's current location
        if let location = locationManager.location {
            region.center = location.coordinate
            region.span = MKCoordinateSpan(latitudeDelta: 0.01, longitudeDelta: 0.01)
            
            // Generate additional resources near user's location
            let nearbyResources = generateNearbyResources(center: location.coordinate)
            resources.append(contentsOf: nearbyResources)
        } else {
            // Fallback to Vienna, VA if current location is not available
            region.center = CLLocationCoordinate2D(latitude: 38.91194, longitude: -77.22244) // Vienna, VA - 1934 Old Gallows Rd
            region.span = MKCoordinateSpan(latitudeDelta: 0.01, longitudeDelta: 0.01)
        }

        isLoading = false
    }
    
    private func generateNearbyResources(center: CLLocationCoordinate2D) -> [Resource] {
        let nearbyResources = [
            Resource(
                name: "Local Community Center",
                type: .communityCenter,
                address: "Near your location",
                latitude: center.latitude + Double.random(in: -0.005...0.005),
                longitude: center.longitude + Double.random(in: -0.005...0.005),
                riskFactor: Double.random(in: 0.1...0.4),
                averageRating: Double.random(in: 3.5...5.0),
                totalRatings: Int.random(in: 10...50),
                incidents: [],
                environmentalFactors: [
                    EnvironmentalFactor(
                        factor: "Lighting",
                        impact: "Well-lit area",
                        riskLevel: 0.2
                    )
                ],
                comments: [
                    Comment(
                        author: "Local Resident",
                        rating: 4,
                        text: "Great community resource nearby",
                        date: Date().addingTimeInterval(-86400 * 2),
                        helpful: 5
                    )
                ],
                trustIndicators: TrustIndicators(
                    communityVerified: true,
                    officialPartnership: false,
                    regularUpdates: true,
                    transparentReporting: true
                )
            ),
            Resource(
                name: "Neighborhood Library",
                type: .library,
                address: "Close to your area",
                latitude: center.latitude + Double.random(in: -0.008...0.008),
                longitude: center.longitude + Double.random(in: -0.008...0.008),
                riskFactor: Double.random(in: 0.05...0.3),
                averageRating: Double.random(in: 4.0...5.0),
                totalRatings: Int.random(in: 15...60),
                incidents: [],
                environmentalFactors: [
                    EnvironmentalFactor(
                        factor: "Security",
                        impact: "Good security presence",
                        riskLevel: 0.1
                    )
                ],
                comments: [
                    Comment(
                        author: "Library User",
                        rating: 5,
                        text: "Safe and quiet environment",
                        date: Date().addingTimeInterval(-86400 * 1),
                        helpful: 8
                    )
                ],
                trustIndicators: TrustIndicators(
                    communityVerified: true,
                    officialPartnership: true,
                    regularUpdates: true,
                    transparentReporting: true
                )
            )
        ]
        
        return nearbyResources
    }

    @MainActor
    private func searchResources() async {
        guard !searchText.isEmpty else { return }

        // Filter resources based on search text
        resources = MockData.sampleResources.filter {
            $0.name.localizedCaseInsensitiveContains(searchText) ||
            $0.address.localizedCaseInsensitiveContains(searchText)
        }
    }
    
    private func centerOnUserLocation() {
        guard let location = locationManager.location else { return }
        
        withAnimation(.easeInOut(duration: 1.0)) {
            region.center = location.coordinate
            region.span = MKCoordinateSpan(latitudeDelta: 0.01, longitudeDelta: 0.01)
        }
    }
    
    private func addIncident(_ incident: Incident) {
        // For now, we'll add the incident to the first resource
        // In a real app, you'd want to match it to the correct resource
        if !resources.isEmpty {
            // This is a simplified approach - in reality you'd need to update the specific resource
            // For now, we'll just add it to the mock data or handle it differently
            print("Incident added: \(incident.type) at \(incident.date)")
        }
    }
    
    private func addComment(_ comment: Comment, to resource: Resource) {
        // For now, we'll just print the comment
        // In a real app, you'd update the specific resource's comments array
        print("Comment added by \(comment.author) for \(resource.name): \(comment.text)")
    }
}

struct SearchBar: View {
    @Binding var text: String
    let onSearchButtonClicked: () -> Void

    var body: some View {
        HStack {
            TextField("Search locations...", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .onSubmit {
                    onSearchButtonClicked()
                }

            Button("Search") {
                onSearchButtonClicked()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

struct FilterBarView: View {
    @Binding var selectedType: ResourceType?

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                FilterChip(
                    title: "All",
                    isSelected: selectedType == nil,
                    action: { selectedType = nil }
                )

                ForEach(ResourceType.allCases, id: \.self) { type in
                    FilterChip(
                        title: type.rawValue,
                        isSelected: selectedType == type,
                        action: { selectedType = selectedType == type ? nil : type }
                    )
                }
            }
            .padding(.horizontal)
        }
        .padding(.vertical, 8)
    }
}

struct FilterChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.medium)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 20)
                        .fill(isSelected ? Color.blue : Color(.systemGray5))
                )
                .foregroundColor(isSelected ? .white : .primary)
        }
    }
}

struct MapView: View {
    let resources: [Resource]
    @Binding var selectedResource: Resource?
    @Binding var region: MKCoordinateRegion
    @StateObject private var locationManager = LocationManager()

    var body: some View {
        Map(coordinateRegion: $region, 
            showsUserLocation: true,
            userTrackingMode: .constant(.none),
            annotationItems: resources) { resource in
            MapAnnotation(coordinate: CLLocationCoordinate2D(
                latitude: resource.latitude ?? 0,
                longitude: resource.longitude ?? 0
            )) {
                ResourceMapPin(resource: resource)
                    .onTapGesture {
                        selectedResource = resource
                    }
            }
        }
        .onAppear {
            locationManager.requestLocationPermission()
        }
    }
}

struct ResourceMapPin: View {
    let resource: Resource

    var riskColor: Color {
        if resource.riskFactor <= 0.3 { return .green }
        else if resource.riskFactor <= 0.6 { return .orange }
        else { return .red }
    }

    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                Circle()
                    .fill(riskColor)
                    .frame(width: 30, height: 30)

                Image(systemName: resource.type.icon)
                    .foregroundColor(.white)
                    .font(.caption)
            }

            Text(resource.name)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundColor(.primary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 80)
        }
    }
}

struct ResourceRowView: View {
    let resource: Resource

    var riskColor: Color {
        if resource.riskFactor <= 0.3 { return .green }
        else if resource.riskFactor <= 0.6 { return .orange }
        else { return .red }
    }

    var riskLabel: String {
        if resource.riskFactor <= 0.3 { return "Low Risk" }
        else if resource.riskFactor <= 0.6 { return "Medium Risk" }
        else { return "High Risk" }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: resource.type.icon)
                    .foregroundColor(resource.type.mapPinColor)
                    .frame(width: 20)

                Text(resource.name)
                    .font(.headline)
                    .fontWeight(.semibold)

                Spacer()

                // Risk Factor Badge
                Text(riskLabel)
                    .font(.caption)
                    .fontWeight(.medium)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(riskColor.opacity(0.2))
                    )
                    .foregroundColor(riskColor)
            }

            Text(resource.address)
                .font(.subheadline)
                .foregroundColor(.secondary)

            HStack {
                // Rating
                HStack(spacing: 2) {
                    ForEach(1...5, id: \.self) { star in
                        Image(systemName: star <= Int(resource.averageRating) ? "star.fill" : "star")
                            .foregroundColor(.yellow)
                            .font(.caption)
                    }
                    Text("(\(resource.totalRatings))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                // Trust Score
                HStack(spacing: 4) {
                    Image(systemName: "checkmark.shield")
                        .foregroundColor(resource.trustIndicators.score > 0.7 ? .green : .orange)
                        .font(.caption)

                    Text(String(format: "%.0f%% Trusted", resource.trustIndicators.score * 100))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            // Recent Incidents Summary
            if !resource.incidents.isEmpty {
                HStack {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundColor(.orange)
                        .font(.caption)

                    Text("\(resource.incidents.count) recent incidents")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Spacer()
                }
            }
        }
        .padding(.vertical, 4)
    }
}


         





struct AddIncidentView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var incidentType = ""
    @State private var description = ""
    @State private var severity = IncidentSeverity.medium
    @State private var selectedResource: Resource?
    @State private var isSubmitting = false
    
    let resources: [Resource]
    let onIncidentAdded: (Incident) -> Void

    let incidentTypes = ["Shooting", "Assault", "Theft", "Vandalism", "Harassment", "Drug Activity", "Other"]
    @State private var incident_ = ""
    @State private var severity_ = ""
    

    var body: some View {
        NavigationView {
            Form {
                Section("Incident Details") {
                    Picker("Type", selection: $incidentType) {
                        ForEach(incidentTypes, id: \.self) { type in
                            Text(type).tag(type)
                        }
                    }
                    .onChange(of: incidentType) { newValue in
                        incidentType = newValue
                    }

                    Picker("Severity", selection: $severity) {
                        ForEach(IncidentSeverity.allCases, id: \.self) { severity in
                            Text(severity.rawValue).tag(severity)
                        }
                    }

                    TextField("Description", text: $description, axis: .vertical)
                        .lineLimit(3...6)
                }

                Section("Location") {
                    Picker("Select Resource", selection: $selectedResource) {
                        ForEach(resources) { resource in
                            Text(resource.name).tag(Optional(resource))
                        }
                    }
                }
            }
            .navigationTitle("Report Incident")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Submit") {
                        Task {
                            await submitIncident()
                            // Fixed: Removed the misplaced Resource creation
                            // This should be handled inside submitIncident() or removed entirely
                        }
                    }
                    .disabled(incidentType.isEmpty || description.isEmpty || isSubmitting)
                }
            }
        }
    }
    @MainActor
    private func submitIncident() async {
        isSubmitting = true
        
        // Create the incident
        let newIncident = Incident(
            type: incidentType,
            date: Date(),
            severity: severity,
            description: description,
            verified: false,
            latitude: selectedResource?.latitude,
            longitude: selectedResource?.longitude
        )
        
        // Add the incident via callback
        onIncidentAdded(newIncident)
        
        dismiss()
        isSubmitting = false
    }
}

struct AddCommentView: View {
    let resources: [Resource]
    let onCommentAdded: (Comment, Resource) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var selectedResource: Resource?
    @State private var rating = 5
    @State private var comment = ""
    @State private var author = ""
    @State private var isSubmitting = false

    var body: some View {
        NavigationView {
            Form {
                resourceSection
                reviewSection
            }
            .navigationTitle("Add Review")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Submit") {
                        Task { await submitComment() }
                    }
                    .disabled(isSubmitDisabled)
                }
            }
        }
    }


    private var resourceSection: some View {
        Section("Resource") {
            Picker("Select Resource", selection: $selectedResource) {
                ForEach(resources) { resource in
                    Text(resource.name).tag(Optional(resource)) // tag as optional
                }
            }
        }
    }

    private var reviewSection: some View {
        Section("Your Review") {
            ratingRow
            TextField("Your Name", text: $author)
            TextField("Write your review...", text: $comment, axis: .vertical)
                .lineLimit(3...6)
        }
    }

    private var ratingRow: some View {
        HStack {
            Text("Rating")
            Spacer()
            HStack(spacing: 2) {
                ForEach(1...5, id: \.self) { star in
                    Image(systemName: star <= rating ? "star.fill" : "star")
                        .foregroundColor(.yellow)
                        .onTapGesture { rating = star }
                }
            }
        }
    }


    private var isSubmitDisabled: Bool {
        selectedResource == nil || comment.isEmpty || author.isEmpty || isSubmitting
    }

    @MainActor
    private func submitComment() async {
        isSubmitting = true
        
        guard let selectedResource = selectedResource else {
            isSubmitting = false
            return
        }
        
        // Create the comment
        let newComment = Comment(
            author: author,
            rating: rating,
            text: comment,
            date: Date(),
            helpful: 0
        )
        
        // Add the comment via callback
        onCommentAdded(newComment, selectedResource)
        
        dismiss()
        isSubmitting = false
    }
}

struct ResourceDetailView: View {
    let resource: Resource
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header
                    ResourceHeaderView(resource: resource)

                    // Trust Indicators
                    TrustIndicatorsView(trustIndicators: resource.trustIndicators)

                    // Recent Incidents
                    if !resource.incidents.isEmpty {
                        IncidentsView(incidents: resource.incidents)
                    }

                    // Environmental Factors
                    if !resource.environmentalFactors.isEmpty {
                        EnvironmentalFactorsView(factors: resource.environmentalFactors)
                    }

                    // Community Comments
                    if !resource.comments.isEmpty {
                        CommentsView(comments: resource.comments)
                    }
                }
                .padding()
            }
            .navigationTitle(resource.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct ResourceHeaderView: View {
    let resource: Resource

    var riskColor: Color {
        if resource.riskFactor <= 0.3 { return .green }
        else if resource.riskFactor <= 0.6 { return .orange }
        else { return .red }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: resource.type.icon)
                    .font(.title2)
                    .foregroundColor(.blue)

                VStack(alignment: .leading) {
                    Text(resource.type.rawValue)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Text(resource.address)
                        .font(.subheadline)
                }

                Spacer()
            }

            // Risk Factor Display
            VStack(alignment: .leading, spacing: 8) {
                Text("Safety Assessment")
                    .font(.headline)

                HStack {
                    Text(String(format: "%.1f%%", resource.riskFactor * 100))
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(riskColor)

                    Text("Risk Level")
                        .foregroundColor(.secondary)

                    Spacer()
                }

                ProgressView(value: resource.riskFactor, total: 1.0)
                    .progressViewStyle(LinearProgressViewStyle(tint: riskColor))
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(.systemGray6))
            )
        }
    }
}

struct TrustIndicatorsView: View {
    let trustIndicators: TrustIndicators

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Trust Indicators")
                .font(.headline)

            VStack(spacing: 8) {
                TrustIndicatorRow(title: "Community Verified", isActive: trustIndicators.communityVerified)
                TrustIndicatorRow(title: "Official Partnership", isActive: trustIndicators.officialPartnership)
                TrustIndicatorRow(title: "Regular Updates", isActive: trustIndicators.regularUpdates)
                TrustIndicatorRow(title: "Transparent Reporting", isActive: trustIndicators.transparentReporting)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
    }
}

struct TrustIndicatorRow: View {
    let title: String
    let isActive: Bool

    var body: some View {
        HStack {
            Image(systemName: isActive ? "checkmark.circle.fill" : "circle")
                .foregroundColor(isActive ? .green : .gray)

            Text(title)
                .font(.subheadline)

            Spacer()
        }
    }
}

struct IncidentsView: View {
    let incidents: [Incident]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent Incidents")
                .font(.headline)

            ForEach(incidents.prefix(3)) { incident in
                SocialJusticeIncidentRowView(incident: incident)
            }

            if incidents.count > 3 {
                Text("+ \(incidents.count - 3) more incidents")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
    }
}

struct SocialJusticeIncidentRowView: View {
    let incident: Incident

    var body: some View {
        HStack {
            Circle()
                .fill(incident.severity.color)
                .frame(width: 8, height: 8)

            VStack(alignment: .leading, spacing: 2) {
                Text(incident.type)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(incident.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)

                Text(incident.date, style: .date)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            if incident.verified {
                Image(systemName: "checkmark.seal")
                    .foregroundColor(.blue)
                    .font(.caption)
            }
        }
    }
}

struct EnvironmentalFactorsView: View {
    let factors: [EnvironmentalFactor]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Environmental Factors")
                .font(.headline)

            ForEach(factors) { factor in
                EnvironmentalFactorRow(factor: factor)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
    }
}

struct EnvironmentalFactorRow: View {
    let factor: EnvironmentalFactor

    var riskColor: Color {
        if factor.riskLevel <= 0.3 { return .green }
        else if factor.riskLevel <= 0.6 { return .orange }
        else { return .red }
    }

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(factor.factor)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(factor.impact)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text(String(format: "%.0f%%", factor.riskLevel * 100))
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(riskColor)
        }
    }
}

struct CommentsView: View {
    let comments: [Comment]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Community Comments")
                .font(.headline)

            ForEach(Array(comments.prefix(3).enumerated()), id: \.element.id) { index, comment in
                CommentRowView(comment: comment, isLast: index == min(2, comments.count - 1))
            }

            if comments.count > 3 {
                Text("+ \(comments.count - 3) more comments")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
    }
}

struct CommentRowView: View {
    let comment: Comment
    let isLast: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(comment.author)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Spacer()

                HStack(spacing: 2) {
                    ForEach(1...5, id: \.self) { star in
                        Image(systemName: star <= comment.rating ? "star.fill" : "star")
                            .foregroundColor(.yellow)
                            .font(.caption)
                    }
                }

                Text(comment.date, style: .date)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Text(comment.text)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(3)

            if comment.helpful > 0 {
                HStack {
                    Image(systemName: "hand.thumbsup")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Text("\(comment.helpful) helpful")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Spacer()
                }
            }
        }
        .padding(.vertical, 4)

        if !isLast {
            Divider()
        }
    }
}

struct MockData {
    static let sampleResources: [Resource] = [
        Resource(
            name: "Vienna Community Center",
            type: .communityCenter,
            address: "120 Cherry St SE, Vienna, VA 22180",
            latitude: 38.9006,
            longitude: -77.2647,
            riskFactor: 0.2,
            averageRating: 4.5,
            totalRatings: 87,
            incidents: [
                Incident(
                    type: "Theft",
                    date: Date().addingTimeInterval(-86400 * 7),
                    severity: .low,
                    description: "Bicycle stolen from bike rack area",
                    verified: true,
                    latitude: 38.9006,
                    longitude: -77.2647
                )
            ],
            environmentalFactors: [
                EnvironmentalFactor(
                    factor: "Lighting Quality",
                    impact: "Well-lit parking and entrance areas",
                    riskLevel: 0.1
                )
            ],
            comments: [
                Comment(
                    author: "Sarah M.",
                    rating: 5,
                    text: "Great facility with excellent security measures. Always feel safe bringing my kids here.",
                    date: Date().addingTimeInterval(-86400 * 3),
                    helpful: 12
                )
            ],
            trustIndicators: TrustIndicators(
                communityVerified: true,
                officialPartnership: true,
                regularUpdates: true,
                transparentReporting: true
            )
        ),
        Resource(
            name: "Vienna Public Library",
            type: .library,
            address: "300 Library Ln, Vienna, VA 22180",
            latitude: 38.8967,
            longitude: -77.2625,
            riskFactor: 0.1,
            averageRating: 4.8,
            totalRatings: 156,
            incidents: [],
            environmentalFactors: [
                EnvironmentalFactor(
                    factor: "Security Cameras",
                    impact: "Comprehensive CCTV coverage",
                    riskLevel: 0.05
                )
            ],
            comments: [
                Comment(
                    author: "Jennifer L.",
                    rating: 5,
                    text: "Wonderful community resource. Very safe environment for studying and reading.",
                    date: Date().addingTimeInterval(-86400 * 2),
                    helpful: 15
                )
            ],
            trustIndicators: TrustIndicators(
                communityVerified: true,
                officialPartnership: true,
                regularUpdates: true,
                transparentReporting: true
            )
        ),
        Resource(
            name: "Vienna Town Green",
            type: .park,
            address: "144 Maple Ave E, Vienna, VA 22180",
            latitude: 38.9012,
            longitude: -77.2641,
            riskFactor: 0.6,
            averageRating: 3.2,
            totalRatings: 43,
            incidents: [
                Incident(
                    type: "Vandalism",
                    date: Date().addingTimeInterval(-86400 * 14),
                    severity: .medium,
                    description: "Playground equipment damaged, graffiti on pavilion",
                    verified: true,
                    latitude: 38.9012,
                    longitude: -77.2641
                ),
                Incident(
                    type: "Assault",
                    date: Date().addingTimeInterval(-86400 * 21),
                    severity: .high,
                    description: "Physical altercation reported near tennis courts",
                    verified: true,
                    latitude: 38.9012,
                    longitude: -77.2641
                )
            ],
            environmentalFactors: [
                EnvironmentalFactor(
                    factor: "Lighting",
                    impact: "Poor lighting in several areas after dark",
                    riskLevel: 0.7
                )
            ],
            comments: [
                Comment(
                    author: "Carlos D.",
                    rating: 2,
                    text: "Park needs better maintenance and security. Avoid after sunset.",
                    date: Date().addingTimeInterval(-86400 * 6),
                    helpful: 9
                )
            ],
            trustIndicators: TrustIndicators(
                communityVerified: false,
                officialPartnership: true,
                regularUpdates: false,
                transparentReporting: true
            )
        ),
        Resource(
            name: "Vienna Health Center",
            type: .healthCenter,
            address: "250 Maple Ave W, Vienna, VA 22180",
            latitude: 38.9034,
            longitude: -77.2689,
            riskFactor: 0.15,
            averageRating: 4.3,
            totalRatings: 92,
            incidents: [
                Incident(
                    type: "Theft",
                    date: Date().addingTimeInterval(-86400 * 10),
                    severity: .low,
                    description: "Car break-in in parking lot",
                    verified: true,
                    latitude: 38.9034,
                    longitude: -77.2689
                )
            ],
            environmentalFactors: [
                EnvironmentalFactor(
                    factor: "Security",
                    impact: "24/7 security presence",
                    riskLevel: 0.1
                ),
                EnvironmentalFactor(
                    factor: "Lighting",
                    impact: "Excellent lighting throughout",
                    riskLevel: 0.05
                )
            ],
            comments: [
                Comment(
                    author: "Dr. Martinez",
                    rating: 5,
                    text: "Very safe facility with excellent security measures.",
                    date: Date().addingTimeInterval(-86400 * 1),
                    helpful: 18
                ),
                Comment(
                    author: "Patient A.",
                    rating: 4,
                    text: "Clean and well-maintained. Feel safe here.",
                    date: Date().addingTimeInterval(-86400 * 4),
                    helpful: 7
                )
            ],
            trustIndicators: TrustIndicators(
                communityVerified: true,
                officialPartnership: true,
                regularUpdates: true,
                transparentReporting: true
            )
        ),
        Resource(
            name: "Vienna Emergency Shelter",
            type: .shelter,
            address: "180 Center St S, Vienna, VA 22180",
            latitude: 38.8998,
            longitude: -77.2712,
            riskFactor: 0.4,
            averageRating: 3.8,
            totalRatings: 34,
            incidents: [
                Incident(
                    type: "Harassment",
                    date: Date().addingTimeInterval(-86400 * 5),
                    severity: .medium,
                    description: "Verbal altercation between residents",
                    verified: true,
                    latitude: 38.8998,
                    longitude: -77.2712
                )
            ],
            environmentalFactors: [
                EnvironmentalFactor(
                    factor: "Crowding",
                    impact: "High occupancy can create tension",
                    riskLevel: 0.5
                ),
                EnvironmentalFactor(
                    factor: "Staffing",
                    impact: "Adequate staff presence",
                    riskLevel: 0.2
                )
            ],
            comments: [
                Comment(
                    author: "Volunteer M.",
                    rating: 4,
                    text: "Good facility but can get crowded. Staff is helpful.",
                    date: Date().addingTimeInterval(-86400 * 2),
                    helpful: 11
                )
            ],
            trustIndicators: TrustIndicators(
                communityVerified: true,
                officialPartnership: true,
                regularUpdates: false,
                transparentReporting: true
            )
        ),
        Resource(
            name: "Vienna Recreation Center",
            type: .communityCenter,
            address: "95 Center St N, Vienna, VA 22180",
            latitude: 38.9056,
            longitude: -77.2654,
            riskFactor: 0.25,
            averageRating: 4.1,
            totalRatings: 67,
            incidents: [],
            environmentalFactors: [
                EnvironmentalFactor(
                    factor: "Lighting",
                    impact: "Good lighting in most areas",
                    riskLevel: 0.2
                ),
                EnvironmentalFactor(
                    factor: "Access Control",
                    impact: "Controlled access during evening hours",
                    riskLevel: 0.1
                )
            ],
            comments: [
                Comment(
                    author: "Family G.",
                    rating: 4,
                    text: "Great place for kids. Generally safe environment.",
                    date: Date().addingTimeInterval(-86400 * 7),
                    helpful: 8
                )
            ],
            trustIndicators: TrustIndicators(
                communityVerified: true,
                officialPartnership: true,
                regularUpdates: true,
                transparentReporting: true
            )
        ),
        Resource(
            name: "Vienna Community Park",
            type: .park,
            address: "200 Park Ave, Vienna, VA 22180",
            latitude: 38.9078,
            longitude: -77.2598,
            riskFactor: 0.35,
            averageRating: 3.9,
            totalRatings: 28,
            incidents: [
                Incident(
                    type: "Drug Activity",
                    date: Date().addingTimeInterval(-86400 * 12),
                    severity: .medium,
                    description: "Suspicious activity reported near restrooms",
                    verified: false,
                    latitude: 38.9078,
                    longitude: -77.2598
                )
            ],
            environmentalFactors: [
                EnvironmentalFactor(
                    factor: "Visibility",
                    impact: "Some areas have limited visibility",
                    riskLevel: 0.4
                )
            ],
            comments: [
                Comment(
                    author: "Local Resident",
                    rating: 3,
                    text: "Nice park but avoid the back areas after dark.",
                    date: Date().addingTimeInterval(-86400 * 3),
                    helpful: 5
                )
            ],
            trustIndicators: TrustIndicators(
                communityVerified: false,
                officialPartnership: true,
                regularUpdates: false,
                transparentReporting: false
            )
        )
    ]
}

