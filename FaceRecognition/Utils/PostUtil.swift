//
//  PostUtil.swift
//  ObjectDetection
//
//  Created by cha on 5/21/25.
//  Copyright © 2025 Y Media Labs. All rights reserved.
//

import Foundation

public class PostUtil {
    
    static func sendMultipartRequest(to urlString: String, with body: [String: Any], completion: @escaping (Result<Data, Error>) -> Void) {
        guard let url = URL(string: urlString) else {
            print("Invalid URL.")
            return
        }
        
        let boundary = "Boundary-\(UUID().uuidString)"
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var formdata = Data()
        for (key, value) in body {
            formdata.append("--\(boundary)\r\n".data(using: .utf8)!)
            formdata.append("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n".data(using: .utf8)!)
            formdata.append("\(value)\r\n".data(using: .utf8)!)
        }
        formdata.append("--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = formdata
        
        if let data = request.httpBody, let dataString = String(data: data, encoding: .utf8) {
            print("리퀘스트 바디!!!\n\(dataString)")
        }
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }

            print(response.debugDescription)
            if let data = data, let responseString = String(data: data, encoding: .utf8) {
                print("응답결과!!!\n\(responseString)")
            }
            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                let statusError = NSError(domain: "HTTPError", code: (response as? HTTPURLResponse)?.statusCode ?? -1, userInfo: nil)
                completion(.failure(statusError))
                return
            }

            if let data = data {
                completion(.success(data))
            } else {
                let noDataError = NSError(domain: "NoData", code: -1, userInfo: nil)
                completion(.failure(noDataError))
            }
        }

        task.resume()
    }
    
    static func sendPostRequest(to urlString: String, with body: [String: Any], completion: @escaping (Result<Data, Error>) -> Void) {
        guard let url = URL(string: urlString) else {
            print("Invalid URL.")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [.fragmentsAllowed])
        } catch {
            completion(.failure(error))
            return
        }
        
        if let data = request.httpBody, let dataString = String(data: data, encoding: .utf8) {
            print("리퀘스트 바디!!!\n\(dataString)")
        }
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }

            print(response.debugDescription)
            if let data = data, let responseString = String(data: data, encoding: .utf8) {
                print("응답결과!!!\n\(responseString)")
            }
            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                let statusError = NSError(domain: "HTTPError", code: (response as? HTTPURLResponse)?.statusCode ?? -1, userInfo: nil)
                completion(.failure(statusError))
                return
            }

            if let data = data {
                completion(.success(data))
            } else {
                let noDataError = NSError(domain: "NoData", code: -1, userInfo: nil)
                completion(.failure(noDataError))
            }
        }

        task.resume()
    }
    
}
