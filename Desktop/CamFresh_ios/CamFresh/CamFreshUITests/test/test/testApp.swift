//
//  testApp.swift
//  test
//
//  Created by RubyLinTu on 2021/6/30.
//

import SwiftUI

@main
struct testApp: App {
    var body: some Scene {
        DocumentGroup(newDocument: testDocument()) { file in
            ContentView(document: file.$document)
        }
    }
}

