//
//  CamFreshApp.swift
//  CamFresh
//
//  Created by RubyLinTu on 2021/6/30.
//

import SwiftUI

@main
struct CamFreshApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
