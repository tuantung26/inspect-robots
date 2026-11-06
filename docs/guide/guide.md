# Tích hợp và đánh giá model mới

Tài liệu này mô tả dữ liệu đầu vào, đầu ra và luồng chạy khi đưa một policy/model mới vào Inspect Robots. Framework cung cấp giao diện chung; adapter vẫn phải chuyển đổi dữ liệu theo đúng yêu cầu của model và robot.

## Các thành phần đầu vào

Một lần đánh giá ghép ba thành phần:

| Thanh phan | Vai tro |
| --- | --- |
| `Task` | Tap cac tinh huong can thu, tieu chi cham va gioi han rollout. |
| `Policy` | Adapter goi model, chuyen quan sat thanh input model va chuyen output model thanh action. |
| `Embodiment` | Adapter robot hoac simulator, tao quan sat va thuc thi action. |

`Task` chứa các `Scene`. Mỗi scene có thể mô tả chỉ dẫn, target, seed và metadata. Số lần lặp (`epochs`) giúp đánh giá độ ổn định của policy trên cùng một scene. Xem [`Task`](../../src/inspect_robots/task.py) và [`Scene`](../../src/inspect_robots/scene.py).

## Dữ liệu policy nhận

Framework gọi `policy.act(observation)` với một [`Observation`](../../src/inspect_robots/types.py), gồm các trường chuẩn:

| Truong | Dinh dang |
| --- | --- |
| `images` | Mapping tu ten camera sang anh NumPy `uint8`, kich thuoc `(H, W, C)`. |
| `state` | Mapping tu ten trang thai sang mang NumPy, vi du `joint_pos`, `eef_pos` hoac `gripper`. |
| `instruction` | Chi dan van ban, co the la `None`. |
| `image_times`, `state_time` | Moc thoi gian cua du lieu neu embodiment cung cap. |
| `extra` | Du lieu bo sung; rollout tu them buoc moi truong va thong tin operator. |

Đây là giao diện của framework, không phải định dạng bắt buộc của model. Policy adapter tự resize/normalize ảnh, tạo tensor, chuẩn hóa state hoặc đổi tên trường theo yêu cầu model. Hợp đồng của policy là [`Policy`](../../src/inspect_robots/policy.py); mô tả camera, state và action space nằm trong [`spaces.py`](../../src/inspect_robots/spaces.py).

### Nếu model cần video tham chiếu

`Observation` không có trường video tham chiếu riêng. Có thể đặt định danh hoặc đường dẫn clip trong metadata của scene, sau đó policy đọc thông tin đó tại `reset(scene)` và nạp/cắt frame theo yêu cầu model. Nếu clip cần được scorer dùng lại, hãy đưa thông tin cần thiết vào `Target.spec` hoặc record metadata; scorer nhận `Record` và `Target`, không nhận trực tiếp `Scene`.

Adapter cần tự xử lý định dạng video, tốc độ frame, đồng bộ thời gian và cách ghép clip tham chiếu với quan sát hiện tại.

## Đầu ra policy

Policy trả về một [`ActionChunk`](../../src/inspect_robots/types.py), gồm một hoặc nhiều `Action`. Mỗi `Action.data` là mảng số NumPy có số chiều phải khớp với action space của embodiment. Ý nghĩa của vector đó được khai báo riêng, ví dụ:

- `joint_pos`: vị trí mục tiêu của các khớp.
- `joint_vel`: vận tốc khớp.
- `joint_delta`: thay đổi vị trí khớp.
- `eef_abs_pose` hoặc `eef_delta_pose`: pose tuyệt đối hoặc thay đổi của đầu công tác.

Action space còn khai báo giới hạn, frame tọa độ, biểu diễn xoay và nhãn từng chiều nếu cần. Adapter phải chuyển output model sang đúng đơn vị, thứ tự khớp, frame và control mode. Không nên giả định framework có thể suy ra các ý nghĩa này từ vector.

Action chunk cho phép model dự đoán nhiều bước từ một lần suy luận. Action manager chọn các bước để thực thi và thời điểm gọi policy lại. Policy có thể phát action kết thúc như `done` hoặc `give_up`, nhưng đây không phải phán xét thành công.

## Embodiment và kết quả mỗi bước

Embodiment khai báo `info`, sau đó cung cấp:

```python
observation = embodiment.reset(scene, seed=seed)
result = embodiment.step(action)
```

`reset()` chuẩn bị scene và trả về quan sát ban đầu. `step()` thực thi một action và trả về `StepResult`: quan sát mới, tùy chọn `reward`, có kết thúc hay không, lý do kết thúc, và `info` tùy chọn cho scorer (ví dụ khoảng cách tới mục tiêu).

Với robot thật, embodiment chịu trách nhiệm kết nối, đọc sensor, gửi lệnh, và đảm bảo nhịp điều khiển an toàn. Rollout không tự chờ theo chu kỳ; embodiment cần tự pacing trong `step()` nếu phần cứng cần điều đó. Xem [`Embodiment`](../../src/inspect_robots/embodiment.py).

## Pipeline đánh giá

```text
Task (Scene + rubric + ngân sách)
              |
              v
Policy và Embodiment khai báo spaces
              |
              v
Kiểm tra tương thích trước rollout
              |
              v
Reset policy và embodiment theo scene/seed
              |
              v
Observation -> policy/model -> ActionChunk
              |
              v
Action manager -> guardrails -> embodiment.step()
              |
              +---- quan sát mới, lặp lại đến khi kết thúc
              |
              v
Record của trial -> grader nếu được cấu hình -> scorer
              |
              v
Gộp điểm qua epochs và scenes -> EvalLog
```

Trước khi chạy, framework kiểm tra kích thước và ngữ nghĩa action, các camera/state policy cần, và khả năng đáp ứng ngân sách của task. Trong rollout, guard có thể giới hạn action hoặc dừng run nếu có lỗi an toàn. Mỗi bước, action chunk, sự kiện và lỗi được ghi vào record.

Sau rollout, grader có thể ghi nhận phán xét của operator hoặc VLM. Scorer đọc record và target để chấm; scorer không điều khiển model. Reducer như `mean` gộp các epoch của một scene, sau đó metric tổng hợp trên các scene có điểm. Nếu muốn chấm độ giống video, benchmark cần định nghĩa scorer và dữ liệu cần thiết, ví dụ cách căn chỉnh thời gian và sai số quỹ đạo; việc đưa video vào model không tự động tạo ra metric đó. Xem [Scoring](scoring.md).

## Đầu ra của lần đánh giá

`eval()` trả về danh sách `EvalLog` và mặc định ghi JSON vào thư mục log. Log chứa:

- Tên task, policy, embodiment; seed và cấu hình chạy.
- Trạng thái, lý do kết thúc và lỗi theo scene/trial.
- Điểm theo rubric, scene và epoch; metric tổng hợp.
- Số bước, thời gian chạy và độ trễ suy luận.
- Tùy chọn: record chi tiết, action chunks, ảnh, transcript và dữ liệu Rerun.

Có thể xem JSON bằng `inspect-robots inspect LOG.json` hoặc tạo báo cáo bằng `inspect-robots view LOG.json`. Cấu trúc log được định nghĩa trong [`log.py`](../../src/inspect_robots/log.py).

## Checklist khi tích hợp model mới

1. Xác định model cần những đầu vào nào và adapter sẽ lấy chúng từ `Observation`, `Scene` hay nguồn ngoài.
2. Định nghĩa action space của embodiment: số chiều, đơn vị, thứ tự, frame, giới hạn và control mode.
3. Viết policy adapter để tiền xử lý input model và chuyển output thành `ActionChunk`.
4. Viết embodiment adapter cho simulator hoặc robot; bắt đầu bằng simulator nếu có thể.
5. Tạo task/scenes và scorer phù hợp với mục tiêu. Tách riêng chấm hoàn thành, an toàn và độ giống demo nếu cần.
6. Chạy kiểm tra tương thích, sau đó thử với seed cố định và xem log trước khi thử trên phần cứng thật.

Framework cung cấp pipeline và giao diện chung, nhưng không tự động viết adapter cho model, robot hoặc video format mới.
