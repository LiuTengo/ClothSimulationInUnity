using System;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class SimulationInfo : MonoBehaviour
{
    public Button restartBtn;
    public Text frameRateText;
    
    private float _updateInterval = 0.25f;//设定更新帧率的时间间隔为1秒  
    private float _accum = .0f;//累积时间  
    private int _frames = 0;//在_updateInterval时间内运行了多少帧  
    private float _timeLeft;
    private string fpsFormat;
    // Start is called before the first frame update
    void Start()
    {
        restartBtn.onClick.AddListener(RestartSimulation);
    }

    private void Update()
    {
        _timeLeft -= Time.deltaTime;
        _accum += Time.timeScale / Time.deltaTime;
        ++_frames;//帧数  

        if (_timeLeft <= 0)
        {
            float fps = _accum / _frames;
            frameRateText.text  = $"{fps:F2}FPS";

            _timeLeft = _updateInterval;
            _accum = .0f;
            _frames = 0;
        }
    }

    private void RestartSimulation()
    {
        Scene scene = SceneManager.GetSceneAt(0);
        SceneManager.LoadScene(scene.name);
    }
}
