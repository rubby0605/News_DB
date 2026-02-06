//
//  ContentView.swift
//  test
//
//  Created by RubyLinTu on 2021/6/30.
//

import SwiftUI

struct ContentView: View {
    @Binding var document: testDocument

    var body: some View {
        TextEditor(text: $document.text)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView(document: .constant(testDocument()))
    }
}
