//
//  ContentView.swift
//  CamFresh
//
//  Created by RubyLinTu on 2021/6/30.
//

import SwiftUI
import CoreData

struct ContentView: View {
    @Environment(\.managedObjectContext) private var viewContext

    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \Item.timestamp, ascending: true)],
        animation: .default)
    private var items: FetchedResults<Item>
    // 使用 UIButton(frame:) 建立一個 UIButton
    myButton = UIButton(
      frame: CGRect(x: 0, y: 0, width: 100, height: 30))

    // 按鈕文字
    myButton.setTitle("按我", forState: .Normal)

    // 按鈕文字顏色
    myButton.setTitleColor(
      UIColor.whiteColor(),
      forState: .Normal)

    // 按鈕是否可以使用
    myButton.enabled = true

    // 按鈕背景顏色
    myButton.backgroundColor = UIColor.darkGrayColor()

    // 按鈕按下後的動作
    myButton.addTarget(
        self,
        action: #selector(ViewController.clickButton),
        forControlEvents: .TouchUpInside)

    // 設置位置並加入畫面
    myButton.center = CGPoint(
        x: fullScreenSize.width * 0.5,
        y: fullScreenSize.height * 0.5)
    self.view.addSubview(myButton)
    var body: some View {
        List {
            ForEach(items) { item in
                Text("Item at \(item.timestamp!, formatter: itemFormatter)")
            }
            .onDelete(perform: deleteItems)
        }
        .toolbar {
            #if os(iOS)
            EditButton()
            #endif

            Button(action: addItem) {
                Label("Add Item", systemImage: "plus")
            }
        }
    }

    private func addItem() {
        withAnimation {
            let newItem = Item(context: viewContext)
            newItem.timestamp = Date()

            do {
                try viewContext.save()
            } catch {
                // Replace this implementation with code to handle the error appropriately.
                // fatalError() causes the application to generate a crash log and terminate. You should not use this function in a shipping application, although it may be useful during development.
                let nsError = error as NSError
                fatalError("Unresolved error \(nsError), \(nsError.userInfo)")
            }
        }
    }

    private func deleteItems(offsets: IndexSet) {
        withAnimation {
            offsets.map { items[$0] }.forEach(viewContext.delete)

            do {
                try viewContext.save()
            } catch {
                // Replace this implementation with code to handle the error appropriately.
                // fatalError() causes the application to generate a crash log and terminate. You should not use this function in a shipping application, although it may be useful during development.
                let nsError = error as NSError
                fatalError("Unresolved error \(nsError), \(nsError.userInfo)")
            }
        }
    }
    func clickButton() {
        // 為基底的 self.view 的底色在黑色與白色兩者間切換
        if self.view.backgroundColor!.isEqual(
          UIColor.whiteColor()) {
            self.view.backgroundColor =
              UIColor.blackColor()
        } else {
            self.view.backgroundColor =
              UIColor.whiteColor()
        }
    }
}

private let itemFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.dateStyle = .short
    formatter.timeStyle = .medium
    return formatter
}()

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView().environment(\.managedObjectContext, PersistenceController.preview.container.viewContext)
    }
}
